using System.Net.Http;
using System.Text;
using System.Text.Json;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class TranslationRefinementService
{
    const int BatchSize = 12;

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

        var result = reviewed.ToList();
        var renderable = reviewed.Where(x => x.Render).ToList();

        for (int offset = 0; offset < renderable.Count; offset += BatchSize)
        {
            token.ThrowIfCancellationRequested();
            var batch = renderable.Skip(offset).Take(BatchSize).ToList();
            progress?.Report($"번역 배치 {offset / BatchSize + 1}/{(renderable.Count + BatchSize - 1) / BatchSize} · {batch.Count}개 블록");

            var translated = await SendBatchAsync(batch, model, token);
            var map = translated.ToDictionary(x => x.Id, x => x.Translation);

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
        var payload = batch.Select(x => new
        {
            id = x.Id,
            corrected_source = x.CorrectedText,
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
                temperature = 0.12,
                num_ctx = 8192,
                num_predict = 2200
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

        var parsed = Parse(content);
        var expected = batch.Select(x => x.Id).OrderBy(x => x).ToArray();
        var actual = parsed.Select(x => x.Id).OrderBy(x => x).ToArray();

        if (!expected.SequenceEqual(actual) ||
            parsed.Any(x => string.IsNullOrWhiteSpace(x.Translation)))
            throw new InvalidOperationException($"번역 결과 검증 실패: {Compact(content)}");

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
1. Use corrected_source as the authoritative source. draft_translation is only a hint and may be replaced completely.
2. Preserve meaning, speaker voice, politeness, emotion, punctuation, names, and recurring terminology.
3. Write concise natural Korean suitable for speech bubbles and captions.
4. Preserve useful visual line structure with [BR]. Use original_region_count as a guide.
5. Do not add explanations, notes, markdown, or reasoning.
6. Return exactly one item for every input id. Never merge, omit, duplicate, or renumber ids.
7. Every translation field must contain finished Korean.

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
            if (!doc.RootElement.TryGetProperty("regions", out var regions) ||
                regions.ValueKind != JsonValueKind.Array)
                return [];

            var result = new List<(int, string)>();
            var used = new HashSet<int>();

            foreach (var region in regions.EnumerateArray())
            {
                if (!region.TryGetProperty("id", out var idEl) ||
                    !idEl.TryGetInt32(out int id) ||
                    !used.Add(id))
                    continue;

                var translation =
                    region.TryGetProperty("translation", out var textEl)
                        ? textEl.GetString()?.Trim() ?? ""
                        : "";

                result.Add((id, translation));
            }

            return result;
        }
        catch
        {
            return [];
        }
    }

    static string Compact(string text)
    {
        var oneLine = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return oneLine.Length <= 500 ? oneLine : oneLine[..500] + "...";
    }
}
