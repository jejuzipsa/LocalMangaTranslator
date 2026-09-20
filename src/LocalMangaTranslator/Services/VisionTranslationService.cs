using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class VisionTranslationService
{
    const int MaxBatchBlocks = 4;
    const double MaxBatchAreaRatio = 0.30;
    const int DefaultVisionImageSide = 1200;

    readonly HttpClient http = new()
    {
        Timeout = TimeSpan.FromMinutes(10)
    };

    sealed record BatchItem(int GlobalId, OcrTextBlock Block);
    sealed record CropPayload(string ImageBase64, int OffsetX, int OffsetY, double Scale);

    public async Task<List<VisionTranslation>> ReviewAndTranslateAsync(
        string imagePath,
        IReadOnlyList<OcrTextBlock> blocks,
        ModelProfile model,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (blocks.Count == 0) return [];

        var batches = BuildBatches(imagePath, blocks);
        progress?.Report($"Vision 입력 · OCR {blocks.Sum(x => x.OriginalRegionCount)}줄 → {blocks.Count}개 블록 → {batches.Count}개 배치");

        var all = new List<VisionTranslation>(blocks.Count);

        for (int i = 0; i < batches.Count; i++)
        {
            token.ThrowIfCancellationRequested();
            progress?.Report($"Vision 배치 {i + 1}/{batches.Count} · {batches[i].Count}개 블록");

            var result = await SendBatchAsync(
                imagePath,
                batches[i],
                model,
                progress,
                token,
                DefaultVisionImageSide);

            all.AddRange(result);
        }

        var ordered = all
            .GroupBy(x => x.Id)
            .Select(x => x.First())
            .OrderBy(x => x.Id)
            .ToList();

        var guarded = ApplyEvidenceGuard(
            ordered,
            progress);

        return await RescueShortSpeechFalseNegativesAsync(
            imagePath,
            guarded,
            model,
            progress,
            token);
    }

    static List<List<BatchItem>> BuildBatches(
        string imagePath,
        IReadOnlyList<OcrTextBlock> blocks)
    {
        var (imageWidth, imageHeight) = GetImageSize(imagePath);
        double pageArea = Math.Max(1, (double)imageWidth * imageHeight);

        var ordered = blocks
            .Select((block, id) => new BatchItem(id, block))
            .OrderBy(x => x.Block.Y + x.Block.H / 2)
            .ThenBy(x => x.Block.X + x.Block.W / 2)
            .ToList();

        var batches = new List<List<BatchItem>>();
        var current = new List<BatchItem>();

        foreach (var item in ordered)
        {
            // Orphan OCR has no detector-confirmed physical owner. Never let
            // it share a Vision crop with valid container-owned dialogue:
            // neighboring text can make a short artifact look like speech,
            // while the orphan can also contaminate a real bubble's review.
            if (item.Block.RegionContainer is null)
            {
                if (current.Count > 0)
                {
                    batches.Add(current);
                    current = [];
                }

                batches.Add([item]);
                continue;
            }

            if (current.Count == 0)
            {
                current.Add(item);
                continue;
            }

            var tentative = current.Append(item).ToList();
            var bounds = GetBounds(tentative.Select(x => x.Block));
            double unionArea = Math.Max(1, (bounds.Right - bounds.X) * (bounds.Bottom - bounds.Y));

            if (current.Count >= MaxBatchBlocks ||
                unionArea / pageArea > MaxBatchAreaRatio)
            {
                batches.Add(current);
                current = [item];
            }
            else
            {
                current.Add(item);
            }
        }

        if (current.Count > 0)
            batches.Add(current);

        return batches;
    }

    async Task<List<VisionTranslation>> SendBatchAsync(
        string imagePath,
        IReadOnlyList<BatchItem> batch,
        ModelProfile model,
        IProgress<string>? progress,
        CancellationToken token,
        int maxImageSide,
        int recoveryAttempt = 0,
        bool shortSpeechRescue = false)
    {
        var crop = await Task.Run(
            () => CreateCropPayload(imagePath, batch.Select(x => x.Block), maxImageSide),
            token);

        var batchBlocks = batch.Select(x => x.Block).ToList();

        var payload = batch.Select((item, localId) => new
        {
            id = localId,
            x = Math.Round((item.Block.X - crop.OffsetX) * crop.Scale, 1),
            y = Math.Round((item.Block.Y - crop.OffsetY) * crop.Scale, 1),
            width = Math.Round(item.Block.W * crop.Scale, 1),
            height = Math.Round(item.Block.H * crop.Scale, 1),
            original_region_count = item.Block.OriginalRegionCount,
            ocr = item.Block.Text,
            secondary_ocr = item.Block.SecondaryOcrText,
            secondary_ocr_source = item.Block.SecondaryOcrSource,
            secondary_ocr_agreement = item.Block.SecondaryOcrAgreement,
            container_kind = item.Block.RegionContainer?.Kind.ToString().ToLowerInvariant(),
            container_score = item.Block.RegionContainer?.Score,
            ocr_confidence = item.Block.Lines.Count == 0
                ? 0
                : Math.Round(item.Block.Lines.Average(x => x.Confidence), 3),
            language = item.Block.Language
        }).ToArray();

        var request = new
        {
            model = model.ModelTag,
            stream = false,
            think = false,
            format = "json",
            options = new
            {
                temperature = recoveryAttempt switch
                {
                    0 => 0.05,
                    1 => 0.10,
                    _ => 0.16
                },
                num_ctx = 8192,
                num_predict = recoveryAttempt switch
                {
                    0 => 1800,
                    1 => 1000,
                    _ => 700
                },
                repeat_penalty = recoveryAttempt == 0 ? 1.08 : 1.14,
                repeat_last_n = 256
            },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = BuildPrompt(
                        payload,
                        shortSpeechRescue),
                    images = new[] { crop.ImageBase64 }
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
        {
            bool contextOverflow =
                response.StatusCode == System.Net.HttpStatusCode.BadRequest &&
                responseText.Contains("exceeds the available context size", StringComparison.OrdinalIgnoreCase);

            bool repeatAbort =
                response.StatusCode == System.Net.HttpStatusCode.InternalServerError &&
                (responseText.Contains("token repeat limit reached", StringComparison.OrdinalIgnoreCase) ||
                 responseText.Contains("prediction aborted", StringComparison.OrdinalIgnoreCase));

            if (contextOverflow && maxImageSide > 680)
            {
                int smallerSide = maxImageSide > 900 ? 850 : 680;
                progress?.Report(
                    $"Vision 컨텍스트 초과 · 이미지 축소 {maxImageSide}→{smallerSide} 재시도");

                return await SendBatchAsync(
                    imagePath,
                    batch,
                    model,
                    progress,
                    token,
                    smallerSide,
                    recoveryAttempt,
                    shortSpeechRescue);
            }

            if (contextOverflow && batch.Count > 1)
            {
                progress?.Report(
                    $"Vision 컨텍스트 초과 · {batch.Count}개 블록을 개별 재시도");

                return await RetryIndividuallyAsync(
                    imagePath,
                    batch,
                    model,
                    progress,
                    token);
            }

            if (repeatAbort && batch.Count > 1)
            {
                progress?.Report(
                    $"Vision 반복 토큰 중단 · {batch.Count}개 블록을 개별 재시도");

                return await RetryIndividuallyAsync(
                    imagePath,
                    batch,
                    model,
                    progress,
                    token);
            }

            if (repeatAbort && recoveryAttempt < 2)
            {
                int nextAttempt = recoveryAttempt + 1;
                int smallerSide = nextAttempt == 1
                    ? Math.Min(maxImageSide, 760)
                    : Math.Min(maxImageSide, 640);

                progress?.Report(
                    $"Vision 반복 토큰 중단 · 단일 블록 복구 재시도 {nextAttempt}/2 " +
                    $"(image={smallerSide}, predict={(nextAttempt == 1 ? 1000 : 700)})");

                return await SendBatchAsync(
                    imagePath,
                    batch,
                    model,
                    progress,
                    token,
                    smallerSide,
                    nextAttempt,
                    shortSpeechRescue);
            }

            if (repeatAbort && batch.Count == 1)
            {
                progress?.Report(
                    $"[vision-keep] id={batch[0].GlobalId} · 반복 토큰 오류가 계속되어 원문 유지");

                return [CreateSafeFallback(batch[0])];
            }

            throw new InvalidOperationException(
                $"Ollama 응답 오류 {(int)response.StatusCode}: {Compact(responseText)}");
        }

        using var outer = JsonDocument.Parse(responseText);
        if (!outer.RootElement.TryGetProperty("message", out var messageEl) ||
            !messageEl.TryGetProperty("content", out var contentEl))
            throw new InvalidOperationException("Ollama 응답에 message.content가 없습니다.");

        var content = contentEl.GetString();
        if (string.IsNullOrWhiteSpace(content))
        {
            if (batch.Count > 1)
                return await RetryIndividuallyAsync(imagePath, batch, model, progress, token);

            progress?.Report(
                $"[vision-keep] id={batch[0].GlobalId} · 단일 블록 Vision 응답이 비어 있어 원문 유지");

            return [CreateSafeFallback(batch[0])];
        }

        var parsed = ParseResult(content, batchBlocks);

        if (!IsCompleteAndUsable(parsed, batchBlocks))
        {
            if (batch.Count > 1)
                return await RetryIndividuallyAsync(imagePath, batch, model, progress, token);

            progress?.Report(
                $"[vision-keep] id={batch[0].GlobalId} · 단일 블록 Vision 결과 검증 실패 · 원문 유지 · " +
                $"{Compact(content)}");

            return [CreateSafeFallback(batch[0])];
        }

        return parsed.Select(x =>
        {
            var sourceItem = batch[x.Id];
            return x with
            {
                Id = sourceItem.GlobalId,
                Source = sourceItem.Block
            };
        }).ToList();
    }

    async Task<List<VisionTranslation>> RescueShortSpeechFalseNegativesAsync(
        string imagePath,
        IReadOnlyList<VisionTranslation> input,
        ModelProfile model,
        IProgress<string>? progress,
        CancellationToken token)
    {
        var result = input.ToList();

        for (int i = 0; i < result.Count; i++)
        {
            token.ThrowIfCancellationRequested();

            var item = result[i];
            if (!ShouldRetryShortSpeech(item))
                continue;

            progress?.Report(
                $"[vision-rescue] id={item.Id} · 짧은 speech 후보 개별 재검수 " +
                $"ocr='{Compact(item.Source.Text)}'");

            try
            {
                var retry = await SendBatchAsync(
                    imagePath,
                    [new BatchItem(item.Id, item.Source)],
                    model,
                    progress,
                    token,
                    900,
                    0,
                    true);

                var rescued = retry.FirstOrDefault();
                if (rescued is null ||
                    !rescued.Render ||
                    !IsRenderableType(rescued.Type) ||
                    string.IsNullOrWhiteSpace(rescued.Translation) ||
                    IsUnsupportedExpansion(rescued))
                {
                    progress?.Report(
                        $"[vision-rescue-keep] id={item.Id} · 짧은 speech 재검수 근거 부족");
                    continue;
                }

                result[i] = rescued;
                progress?.Report(
                    $"[vision-rescue-ok] id={item.Id} · " +
                    $"'{Compact(rescued.CorrectedText)}' → '{Compact(rescued.Translation)}'");
            }
            catch (Exception ex) when (
                ex is InvalidOperationException or
                HttpRequestException or
                TaskCanceledException)
            {
                progress?.Report(
                    $"[vision-rescue-keep] id={item.Id} · {Compact(ex.Message)}");
            }
        }

        return result;
    }

    static bool ShouldRetryShortSpeech(
        VisionTranslation item)
    {
        if (item.Render ||
            item.Source.RegionContainer?.Kind !=
                ContainerCandidateKind.Speech ||
            item.Source.OriginalRegionCount != 1 ||
            item.Source.Lines.Count == 0)
        {
            return false;
        }

        int meaningful =
            MeaningfulLength(
                item.Source.Text);

        if (meaningful is < 1 or > 4)
            return false;

        double confidence =
            item.Source.Lines.Average(
                x => x.Confidence);

        return confidence < 0.78;
    }

    async Task<List<VisionTranslation>> RetryIndividuallyAsync(
        string imagePath,
        IReadOnlyList<BatchItem> batch,
        ModelProfile model,
        IProgress<string>? progress,
        CancellationToken token)
    {
        var results = new List<VisionTranslation>();

        foreach (var item in batch)
        {
            token.ThrowIfCancellationRequested();

            try
            {
                var one = await SendBatchAsync(
                    imagePath,
                    [item],
                    model,
                    progress,
                    token,
                    760);

                results.AddRange(one);
            }
            catch (InvalidOperationException ex)
            {
                // 한 블록의 Vision 실패 때문에 전체 페이지/전체 큐를 중단하지 않는다.
                // 분류가 불확실하므로 해당 블록은 보수적으로 원문 유지한다.
                progress?.Report(
                    $"[vision-keep] id={item.GlobalId} · 개별 검수 실패 · {Compact(ex.Message)}");

                results.Add(CreateSafeFallback(item));
            }
        }

        return results;
    }

    static VisionTranslation CreateSafeFallback(BatchItem item)
        => new(
            item.GlobalId,
            item.Block,
            item.Block.Text,
            "",
            "other",
            false);

    static List<VisionTranslation> ApplyEvidenceGuard(
        IReadOnlyList<VisionTranslation> input,
        IProgress<string>? progress)
    {
        var result = new List<VisionTranslation>(
            input.Count);

        foreach (var item in input)
        {
            if (!item.Render ||
                !IsUnsupportedExpansion(item))
            {
                result.Add(item);
                continue;
            }

            progress?.Report(
                $"[vision-keep] id={item.Id} · OCR 근거보다 교정문이 과도하게 확장되어 원문 유지 " +
                $"('{Compact(item.Source.Text)}' → '{Compact(item.CorrectedText)}')");

            result.Add(new VisionTranslation(
                item.Id,
                item.Source,
                item.Source.Text,
                "",
                "other",
                false));
        }

        return result;
    }

    static bool IsUnsupportedExpansion(
        VisionTranslation item)
    {
        var source = item.Source;

        if (source.OriginalRegionCount > 1 ||
            source.Lines.Count == 0)
            return false;

        int sourceLength =
            MeaningfulLength(
                source.Text);

        int correctedLength =
            MeaningfulLength(
                item.CorrectedText);

        double confidence =
            source.Lines.Average(
                x => x.Confidence);

        // 한 줄짜리 저신뢰 OCR 한두 글자는 주변 대사의 일부일 가능성이 높다.
        // Vision이 이를 긴 독립 문장으로 재구성하더라도 공간적 근거가 없으므로
        // 새 번역 Unit을 만들지 않는다. "NO.", "RUN!" 같은 실제 짧은 대사는
        // 교정문 길이가 원문과 비슷하므로 이 규칙에 걸리지 않는다.
        if (sourceLength <= 3 &&
            confidence < 0.72 &&
            correctedLength >= Math.Max(
                8,
                sourceLength * 3 + 3))
        {
            return true;
        }

        // OCR이 거의 읽지 못한 한 줄을 Vision이 지나치게 크게 확장하는 경우도
        // 같은 원칙으로 보수적으로 원문을 유지한다.
        if (sourceLength <= 6 &&
            confidence < 0.50 &&
            correctedLength >= Math.Max(
                18,
                sourceLength * 4))
        {
            return true;
        }

        return false;
    }

    static int MeaningfulLength(
        string? text)
        => string.IsNullOrWhiteSpace(text)
            ? 0
            : text.Count(char.IsLetterOrDigit);

    static bool IsCompleteAndUsable(
        IReadOnlyList<VisionTranslation> result,
        IReadOnlyList<OcrTextBlock> blocks)
    {
        if (result.Count != blocks.Count)
            return false;

        var ids = result.Select(x => x.Id).OrderBy(x => x).ToArray();
        for (int i = 0; i < blocks.Count; i++)
            if (ids[i] != i)
                return false;

        foreach (var item in result)
        {
            if (!item.Render)
                continue;

            if (string.IsNullOrWhiteSpace(item.Translation))
                return false;

            var source = blocks[item.Id];
            bool sourceUsesLatin = source.Text.Any(c => c is >= 'A' and <= 'Z' or >= 'a' and <= 'z');
            bool hasHangul = item.Translation.Any(c => c is >= '\uAC00' and <= '\uD7A3');

            // "NO.", "NEVER.", "RUN!" 같은 초단문도 반드시 한국어 결과여야 한다.
            if (sourceUsesLatin && IsRenderableType(item.Type) && !hasHangul)
                return false;

            if (sourceUsesLatin &&
                NormalizeComparable(item.Translation) == NormalizeComparable(source.Text))
                return false;
        }

        return true;
    }

    static string BuildPrompt(
        object payload,
        bool shortSpeechRescue = false)
    {
        var json = JsonSerializer.Serialize(payload);

        string rescueInstructions =
            shortSpeechRescue
                ? """
SHORT-SPEECH RESCUE MODE:
- This block is inside a page detector-confirmed speech bubble.
- The primary OCR is very short and low-confidence, so strings such as "E7" may actually be a visible interjection such as "Eh?" or "Huh?".
- Inspect the attached crop carefully and correct only glyphs that are visibly supported.
- Do not classify the block as other merely because the OCR text looks nonsensical.
- If the visible content is speech, return dialogue/thought with render=true and a concise Korean translation.
- If the crop genuinely does not support readable speech, keep render=false rather than inventing text.

"""
                : "";

        const string instructions = """
TARGET LANGUAGE: Korean (한국어).
You are a professional comic and manga translator. The attached image crop and OCR blocks belong to the same comic page.

NON-NEGOTIABLE RULES:
1. Return Korean only in every translation field. Never leave ordinary English/Japanese source fragments untranslated.
2. Use the image, neighboring blocks, character emotion, scene context, and surrounding dialogue to resolve meaning.
3. OCR is only a draft. Correct OCR against the visible image before translating.
3a. secondary_ocr, when present, comes from an independent manga OCR. Treat it as separate transcription evidence, not as truth. If primary OCR and secondary OCR disagree, inspect the image and choose only what is visibly supported. Never concatenate competing OCR guesses.
4. Each input block may contain several OCR lines from one speech bubble or caption. Reconstruct them as ONE coherent utterance before translating.
5. Preserve meaning first. Do not invent facts, relationships, motives, names, or details that are not supported by the page.
6. Make the final Korean natural and concise while preserving speaker voice, politeness, emotional force, punctuation, and comic rhythm.
7. Keep names and recurring terms consistent. Transliterate common proper nouns naturally into Korean when appropriate.
8. Classify each block as one of: dialogue, thought, caption, sign, sfx, logo, background, other.
9. Set render=true ONLY for dialogue, thought, and narrative caption. Set render=false for sign, sfx, logo, background, and other artwork text.
10. Very short speech is still dialogue. Words such as "NO.", "YES.", "WAIT!", "RUN!", "NEVER." must never be dropped merely because they are short.
11. For render=true, translation MUST be a finished Korean translation, never unchanged source text.
12. Before finalizing each translation, silently check three things: literal fidelity, natural Korean speech/caption style, and brevity for a speech balloon. Do not output the checks.
13. Avoid stiff translationese. Use the shortest natural Korean wording that preserves the exact intent, relationship, register, emotion, and emphasis.
14. Neighboring input blocks are context-only. They may guide names, honorifics, pronouns, and speaker tone, but their words, facts, numbers, actions, and clauses MUST NOT be copied into another id. Each id owns only the visible text inside its own OCR block/container.
15. Preserve useful visual line structure with [BR]. Use original_region_count as a guide:
    - 1 line: normally no [BR]
    - 2 lines: normally one [BR]
    - 3+ lines: use readable balanced breaks close to the original line count
    Break at natural phrase boundaries; never strand a weak particle by itself.
16. Keep exactly one output object for every input id. Do not merge, omit, duplicate, or renumber ids.
17. Do not output reasoning, notes, markdown, or commentary. Output one JSON object only.

OUTPUT SCHEMA:
{
  "regions": [
    {
      "id": 0,
      "corrected": "exact corrected source text for this block",
      "translation": "natural Korean[BR]with optional visual breaks",
      "type": "dialogue",
      "render": true
    }
  ]
}

INPUT BLOCKS:
""";

        return rescueInstructions +
               instructions +
               Environment.NewLine +
               json;
    }

    static List<VisionTranslation> ParseResult(
        string content,
        IReadOnlyList<OcrTextBlock> blocks)
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

            var result = new List<VisionTranslation>();
            var used = new HashSet<int>();

            foreach (var region in regions.EnumerateArray())
            {
                if (!region.TryGetProperty("id", out var idEl) ||
                    !idEl.TryGetInt32(out int id) ||
                    id < 0 ||
                    id >= blocks.Count ||
                    !used.Add(id))
                    continue;

                string corrected =
                    region.TryGetProperty("corrected", out var correctedEl)
                        ? correctedEl.GetString()?.Trim() ?? ""
                        : "";

                string translation =
                    region.TryGetProperty("translation", out var translationEl)
                        ? translationEl.GetString()?.Trim() ?? ""
                        : "";

                string type =
                    region.TryGetProperty("type", out var typeEl)
                        ? NormalizeType(typeEl.GetString())
                        : "dialogue";

                bool requestedRender =
                    !region.TryGetProperty("render", out var renderEl) ||
                    renderEl.ValueKind != JsonValueKind.False;

                bool render =
                    requestedRender &&
                    IsRenderableType(type);

                result.Add(new VisionTranslation(
                    id,
                    blocks[id],
                    string.IsNullOrWhiteSpace(corrected) ? blocks[id].Text : corrected,
                    translation,
                    type,
                    render));
            }

            return result.OrderBy(x => x.Id).ToList();
        }
        catch
        {
            return [];
        }
    }

    static string NormalizeType(string? value)
    {
        return value?.Trim().ToLowerInvariant() switch
        {
            "dialogue" => "dialogue",
            "thought" => "thought",
            "caption" => "caption",
            "sign" => "sign",
            "sfx" => "sfx",
            "logo" => "logo",
            "background" => "background",
            "other" => "other",
            _ => "dialogue"
        };
    }

    static bool IsRenderableType(string? type)
        => type is "dialogue" or "thought" or "caption";

    static string NormalizeComparable(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return "";

        return new string(
            text.Where(char.IsLetterOrDigit)
                .Select(char.ToLowerInvariant)
                .ToArray());
    }

    static CropPayload CreateCropPayload(
        string imagePath,
        IEnumerable<OcrTextBlock> blocks,
        int maxImageSide)
    {
        using var input = File.OpenRead(imagePath);
        var frame = BitmapDecoder.Create(
            input,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad).Frames[0];

        var bounds = GetBounds(blocks);
        double width = Math.Max(1, bounds.Right - bounds.X);
        double height = Math.Max(1, bounds.Bottom - bounds.Y);

        double padX = Math.Clamp(width * 0.16, 40, 140);
        double padY = Math.Clamp(height * 0.20, 40, 150);

        int x = Math.Max(0, (int)Math.Floor(bounds.X - padX));
        int y = Math.Max(0, (int)Math.Floor(bounds.Y - padY));
        int right = Math.Min(frame.PixelWidth, (int)Math.Ceiling(bounds.Right + padX));
        int bottom = Math.Min(frame.PixelHeight, (int)Math.Ceiling(bounds.Bottom + padY));
        int cropWidth = Math.Max(1, right - x);
        int cropHeight = Math.Max(1, bottom - y);

        BitmapSource source = new CroppedBitmap(
            frame,
            new Int32Rect(x, y, cropWidth, cropHeight));

        double longSide = Math.Max(cropWidth, cropHeight);
        double scale = 1.0;

        if (longSide > maxImageSide)
            scale = maxImageSide / longSide;
        else if (longSide < 650)
            scale = Math.Min(1.6, 820.0 / longSide);

        if (Math.Abs(scale - 1.0) > 0.01)
            source = new TransformedBitmap(source, new ScaleTransform(scale, scale));

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(source));

        using var output = new MemoryStream();
        encoder.Save(output);

        return new CropPayload(
            Convert.ToBase64String(output.ToArray()),
            x,
            y,
            scale);
    }

    static (int Width, int Height) GetImageSize(string imagePath)
    {
        using var input = File.OpenRead(imagePath);
        var frame = BitmapDecoder.Create(
            input,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad).Frames[0];

        return (frame.PixelWidth, frame.PixelHeight);
    }

    static (double X, double Y, double Right, double Bottom) GetBounds(
        IEnumerable<OcrTextBlock> blocks)
    {
        var list = blocks.ToList();
        if (list.Count == 0) return (0, 0, 1, 1);

        return (
            list.Min(x => x.X),
            list.Min(x => x.Y),
            list.Max(x => x.X + x.W),
            list.Max(x => x.Y + x.H));
    }

    static string Compact(string text)
    {
        var oneLine = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return oneLine.Length <= 500 ? oneLine : oneLine[..500] + "...";
    }
}
