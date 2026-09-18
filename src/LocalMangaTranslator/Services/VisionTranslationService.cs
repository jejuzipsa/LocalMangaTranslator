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
                token,
                DefaultVisionImageSide);

            all.AddRange(result);
        }

        return all
            .GroupBy(x => x.Id)
            .Select(x => x.First())
            .OrderBy(x => x.Id)
            .ToList();
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
        CancellationToken token,
        int maxImageSide)
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
                temperature = 0.05,
                num_ctx = 8192,
                num_predict = 1800
            },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = BuildPrompt(payload),
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

            if (contextOverflow && maxImageSide > 680)
            {
                int smallerSide = maxImageSide > 900 ? 850 : 680;
                return await SendBatchAsync(imagePath, batch, model, token, smallerSide);
            }

            if (contextOverflow && batch.Count > 1)
                return await RetryIndividuallyAsync(imagePath, batch, model, token);

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
                return await RetryIndividuallyAsync(imagePath, batch, model, token);

            throw new InvalidOperationException("Vision LLM 응답이 비어 있습니다.");
        }

        var parsed = ParseResult(content, batchBlocks);

        if (!IsCompleteAndUsable(parsed, batchBlocks))
        {
            if (batch.Count > 1)
                return await RetryIndividuallyAsync(imagePath, batch, model, token);

            throw new InvalidOperationException(
                $"Vision LLM 결과 검증 실패: {Compact(content)}");
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

    async Task<List<VisionTranslation>> RetryIndividuallyAsync(
        string imagePath,
        IReadOnlyList<BatchItem> batch,
        ModelProfile model,
        CancellationToken token)
    {
        var results = new List<VisionTranslation>();

        foreach (var item in batch)
        {
            token.ThrowIfCancellationRequested();
            var one = await SendBatchAsync(imagePath, [item], model, token, 760);
            results.AddRange(one);
        }

        return results;
    }

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

    static string BuildPrompt(object payload)
    {
        var json = JsonSerializer.Serialize(payload);

        const string instructions = """
TARGET LANGUAGE: Korean (한국어).
You are a professional comic and manga translator. The attached image crop and OCR blocks belong to the same comic page.

NON-NEGOTIABLE RULES:
1. Return Korean only in every translation field. Never leave ordinary English/Japanese source fragments untranslated.
2. Use the image, neighboring blocks, character emotion, scene context, and surrounding dialogue to resolve meaning.
3. OCR is only a draft. Correct OCR against the visible image before translating.
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
14. Read neighboring input blocks as local page context so recurring names, honorifics, pronouns, and speaker tone stay consistent, but never merge separate ids.
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

        return instructions + Environment.NewLine + json;
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
