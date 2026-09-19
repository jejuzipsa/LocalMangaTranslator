using System.Net.Http;
using System.Text;
using System.Text.Json;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class TranslationRefinementService
{
    // Final translation is deliberately isolated per translation unit.
    // 0011 proved that even a well-formed multi-id JSON response can move
    // propositional content from one neighboring bubble into another.
    // Text-only single-unit calls are slower, but preserve the container
    // boundary as a hard semantic boundary.
    const int BatchSize = 1;

    readonly HttpClient http = new()
    {
        Timeout = TimeSpan.FromMinutes(10)
    };

    public async Task<List<VisionTranslation>> TranslateAsync(
        IReadOnlyList<VisionTranslation> reviewed,
        ModelProfile model,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (reviewed.Count == 0) return [];

        // Vision 검수 단계의 번역을 항상 안전한 fallback으로 보존한다.
        var result = reviewed.ToList();
        var renderable = reviewed
            .Where(x => x.Render && IsRenderableType(x.Type))
            .ToList();

        for (int offset = 0; offset < renderable.Count; offset += BatchSize)
        {
            token.ThrowIfCancellationRequested();

            var batch = renderable.Skip(offset).Take(BatchSize).ToList();
            int batchIndex = offset / BatchSize + 1;
            int batchCount = (renderable.Count + BatchSize - 1) / BatchSize;

            progress?.Report($"번역 배치 {batchIndex}/{batchCount} · {batch.Count}개 블록");

            var map = new Dictionary<int, string>();

            try
            {
                var translated = await SendBatchAsync(batch, model, token);
                foreach (var item in translated)
                {
                    var source = batch.FirstOrDefault(x => x.Id == item.Id);
                    if (source is not null &&
                        IsUsableTranslation(source, item.Translation))
                        map[item.Id] = item.Translation;
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // 배치 전체가 실패해도 바로 페이지 전체를 중단하지 않는다.
                progress?.Report($"번역 배치 응답 문제 · 누락 블록만 재시도: {Compact(ex.Message)}");
            }

            var missing = batch
                .Where(x => !map.TryGetValue(x.Id, out var text) || string.IsNullOrWhiteSpace(text))
                .ToList();

            foreach (var item in missing)
            {
                token.ThrowIfCancellationRequested();
                progress?.Report($"번역 누락 ID {item.Id} · 개별 재시도");

                try
                {
                    var one = await SendBatchAsync([item], model, token);
                    var recovered = one.FirstOrDefault(x =>
                        x.Id == item.Id && !string.IsNullOrWhiteSpace(x.Translation));

                    if (IsUsableTranslation(item, recovered.Translation))
                    {
                        map[item.Id] = recovered.Translation;
                        continue;
                    }
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    progress?.Report($"번역 ID {item.Id} 재시도 실패 · Vision 번역 유지: {Compact(ex.Message)}");
                }

                // reviewed의 Translation은 Vision 검수 단계에서 이미 검증된 한국어 번역이다.
                // 최종 번역 모델이 형식을 깨뜨려도 이 값을 유지하면 작업 전체는 계속 진행된다.
                if (!string.IsNullOrWhiteSpace(item.Translation))
                    map[item.Id] = item.Translation;
            }

            for (int i = 0; i < result.Count; i++)
            {
                if (map.TryGetValue(result[i].Id, out var text) && !string.IsNullOrWhiteSpace(text))
                    result[i] = result[i] with { Translation = text };
            }
        }

        return result.OrderBy(x => x.Id).ToList();
    }

    async Task<List<(int Id, string Translation)>> SendBatchAsync(
        IReadOnlyList<VisionTranslation> batch,
        ModelProfile model,
        CancellationToken token)
    {
        var payload = batch.Select((x, index) => new
        {
            id = x.Id,
            sequence = index,
            corrected_source = x.CorrectedText,
            source_language = x.Source.Language,
            type = x.Type,
            original_region_count = x.Source.OriginalRegionCount,
            draft_translation = x.Translation
        }).ToArray();

        var request = new
        {
            model = model.ModelTag,
            stream = false,
            think = false,
            format = "json",
            options = new
            {
                temperature = 0.08,
                num_ctx = 8192,
                num_predict = 1800
            },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = BuildPrompt(payload)
                }
            }
        };

        using var message = new HttpRequestMessage(
            HttpMethod.Post,
            $"{model.ApiBase.TrimEnd('/')}/api/chat")
        {
            Content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json")
        };

        using var response = await http.SendAsync(message, token);
        var responseText = await response.Content.ReadAsStringAsync(token);

        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException(
                $"번역 모델 응답 오류 {(int)response.StatusCode}: {Compact(responseText)}");

        using var outer = JsonDocument.Parse(responseText);
        if (!outer.RootElement.TryGetProperty("message", out var messageEl) ||
            !messageEl.TryGetProperty("content", out var contentEl))
            throw new InvalidOperationException("번역 모델 응답에 message.content가 없습니다.");

        var content = contentEl.GetString();
        if (string.IsNullOrWhiteSpace(content))
            throw new InvalidOperationException("번역 모델 응답이 비어 있습니다.");

        var expectedIds = batch.Select(x => x.Id).ToHashSet();
        var parsed = Parse(content)
            .Where(x => expectedIds.Contains(x.Id) && !string.IsNullOrWhiteSpace(x.Translation))
            .GroupBy(x => x.Id)
            .Select(x => x.First())
            .ToList();

        // 부분 응답도 살려서 반환한다. 누락 ID는 호출부에서 개별 재시도한다.
        if (parsed.Count == 0)
            throw new InvalidOperationException($"번역 결과를 읽지 못했습니다: {Compact(content)}");

        return parsed;
    }

    static string BuildPrompt(object payload)
    {
        var json = JsonSerializer.Serialize(payload);

        const string instructions = """
TARGET LANGUAGE: Korean (한국어).
You are the final translation stage for a comic/manga localization pipeline.
The Vision review stage has already corrected OCR against the image. Translate the corrected source into natural Korean.

RULES:
1. Use corrected_source as the authoritative source. draft_translation is only a draft and may be replaced completely.
2. Treat corrected_source for this id as a protected semantic boundary. Translate ONLY information present in that corrected_source. Never import a clause, fact, number, action, object, or conclusion from another bubble/caption.
3. Preserve exact meaning, intent, speaker attitude, politeness level, emotion, emphasis, punctuation, names, and recurring terminology. Do not invent information. If draft_translation contains meaning that is absent from corrected_source, discard that extra meaning completely.
4. Produce idiomatic Korean that sounds written for a Korean comic, not like a literal machine translation. Remove English word order and stiff translationese.
5. Prefer concise speech-bubble wording. If two Korean phrasings mean the same thing, choose the shorter and more natural one.
6. Preserve character voice. Casual, rough, formal, old-fashioned, sarcastic, threatening, hesitant, or intimate speech should remain distinct when supported by the source.
7. Very short dialogue is important. Translate standalone speech such as "NO.", "YES.", "WAIT!", "RUN!", "NEVER." into natural Korean; never return it unchanged.
8. Silently perform three checks before emitting each result: (a) fidelity to source, (b) natural Korean, (c) balloon brevity. Output only the final wording.
9. Preserve useful visual line structure with [BR]. Use original_region_count only as a layout hint, not as a reason to split grammar unnaturally.
10. Every translation field must be finished Korean. Do not leave ordinary English/Japanese source fragments untranslated. Proper names should be transliterated naturally when appropriate.
11. Do not add explanations, notes, markdown, reasoning, or extra fields.
12. Return exactly one item for the input id. Never merge, omit, duplicate, renumber, continue, or complete the sentence using text from another region.
13. ALL output items MUST be inside the regions array. Never place id or translation at the root object.
14. Output only id and translation for each region. Do not echo sequence, source_language, original_region_count, corrected_source, type, or draft_translation.
15. Keep the JSON structure valid until every input id has been emitted.

OUTPUT:
{
  "regions": [
    { "id": 0, "translation": "완성된 한국어 번역" }
  ]
}

INPUT:
""";

        return instructions + Environment.NewLine + json;
    }

    static List<(int Id, string Translation)> Parse(string content)
    {
        try
        {
            var normalized = content.Trim();
            var fence = new string((char)96, 3);

            if (normalized.StartsWith(fence))
            {
                int firstNewLine = normalized.IndexOf('\n');
                if (firstNewLine >= 0)
                    normalized = normalized[(firstNewLine + 1)..];

                int lastFence = normalized.LastIndexOf(fence, StringComparison.Ordinal);
                if (lastFence >= 0)
                    normalized = normalized[..lastFence].Trim();
            }

            using var doc = JsonDocument.Parse(normalized);
            var result = new List<(int, string)>();
            var used = new HashSet<int>();

            if (doc.RootElement.TryGetProperty("regions", out var regions) &&
                regions.ValueKind == JsonValueKind.Array)
            {
                foreach (var region in regions.EnumerateArray())
                    TryAdd(region, result, used);
            }

            // Gemma가 마지막 항목을 regions 배열 밖(root)에 내보내는 사례를 회수한다.
            TryAdd(doc.RootElement, result, used);

            return result;
        }
        catch
        {
            return [];
        }
    }

    static void TryAdd(
        JsonElement element,
        List<(int Id, string Translation)> result,
        HashSet<int> used)
    {
        if (element.ValueKind != JsonValueKind.Object ||
            !element.TryGetProperty("id", out var idEl) ||
            !idEl.TryGetInt32(out int id) ||
            !element.TryGetProperty("translation", out var textEl))
            return;

        var translation = textEl.GetString()?.Trim() ?? "";
        if (string.IsNullOrWhiteSpace(translation) || !used.Add(id))
            return;

        result.Add((id, translation));
    }

    static bool IsRenderableType(string? type)
        => type is "dialogue" or "thought" or "caption";

    static bool IsUsableTranslation(
        VisionTranslation source,
        string? translation)
    {
        if (string.IsNullOrWhiteSpace(translation))
            return false;

        string text = translation.Trim();

        bool sourceUsesLatin =
            source.CorrectedText.Any(c =>
                c is >= 'A' and <= 'Z' or >= 'a' and <= 'z');

        if (!sourceUsesLatin)
            return true;

        bool hasHangul =
            text.Any(c => c is >= '\uAC00' and <= '\uD7A3');

        // 영어 대사/캡션이 그대로 남는 결과를 막는다.
        if (IsRenderableType(source.Type) && !hasHangul)
            return false;

        if (NormalizeComparable(text) ==
            NormalizeComparable(source.CorrectedText))
            return false;

        int latin = text.Count(c =>
            c is >= 'A' and <= 'Z' or >= 'a' and <= 'z');

        int hangul = text.Count(c =>
            c is >= '\uAC00' and <= '\uD7A3');

        // 고유명사 몇 글자는 허용하되, 결과 대부분이 영어라면 실패로 보고
        // 개별 재시도 후 Vision 단계 번역으로 fallback한다.
        if (latin >= 4 &&
            latin > Math.Max(4, hangul * 2))
            return false;

        return true;
    }

    static string NormalizeComparable(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return "";

        return new string(
            text.Where(char.IsLetterOrDigit)
                .Select(char.ToLowerInvariant)
                .ToArray());
    }

    static string Compact(string text)
    {
        var oneLine = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return oneLine.Length <= 500 ? oneLine : oneLine[..500] + "...";
    }
}
