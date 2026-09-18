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
    const int MaxBatchRegions = 6;
    const double MaxBatchAreaRatio = 0.24;
    const int DefaultVisionImageSide = 1100;

    readonly HttpClient http = new()
    {
        Timeout = TimeSpan.FromMinutes(10)
    };

    sealed record BatchItem(int GlobalId, OcrLine Line);
    sealed record CropPayload(string ImageBase64, int OffsetX, int OffsetY, double Scale);

    public async Task<List<VisionTranslation>> ReviewAndTranslateAsync(
        string imagePath,
        IReadOnlyList<OcrLine> ocrLines,
        ModelProfile model,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (ocrLines.Count == 0) return [];

        var batches = BuildBatches(imagePath, ocrLines);
        progress?.Report($"Vision 입력을 {batches.Count}개 배치로 분할 · 총 {ocrLines.Count}개 영역");

        var all = new List<VisionTranslation>(ocrLines.Count);

        for (int i = 0; i < batches.Count; i++)
        {
            token.ThrowIfCancellationRequested();
            progress?.Report($"Vision 배치 {i + 1}/{batches.Count} · {batches[i].Count}개 영역");

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

    static List<List<BatchItem>> BuildBatches(string imagePath, IReadOnlyList<OcrLine> lines)
    {
        var (imageWidth, imageHeight) = GetImageSize(imagePath);
        double pageArea = Math.Max(1, (double)imageWidth * imageHeight);

        var ordered = lines
            .Select((line, id) => new BatchItem(id, line))
            .OrderBy(x => x.Line.Y + x.Line.H / 2)
            .ThenBy(x => x.Line.X + x.Line.W / 2)
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
            var bounds = GetBounds(tentative.Select(x => x.Line));
            double unionArea = Math.Max(1, (bounds.Right - bounds.X) * (bounds.Bottom - bounds.Y));
            bool tooWide = unionArea / pageArea > MaxBatchAreaRatio;
            bool tooMany = current.Count >= MaxBatchRegions;

            if (tooWide || tooMany)
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
            () => CreateCropPayload(imagePath, batch.Select(x => x.Line), maxImageSide),
            token);

        var batchLines = batch.Select(x => x.Line).ToList();
        var ocrPayload = batch.Select((item, localId) => new
        {
            id = localId,
            x = Math.Round((item.Line.X - crop.OffsetX) * crop.Scale, 1),
            y = Math.Round((item.Line.Y - crop.OffsetY) * crop.Scale, 1),
            width = Math.Round(item.Line.W * crop.Scale, 1),
            height = Math.Round(item.Line.H * crop.Scale, 1),
            ocr = item.Line.Text,
            confidence = Math.Round(item.Line.Confidence, 3),
            language = item.Line.Language
        }).ToArray();

        var prompt = BuildPrompt(ocrPayload);

        var request = new
        {
            model = model.ModelTag,
            stream = false,
            think = false,
            format = "json",
            options = new
            {
                temperature = 0.1,
                num_ctx = 8192,
                num_predict = 1400
            },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = prompt,
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

            if (contextOverflow && maxImageSide > 640)
            {
                int smallerSide = maxImageSide > 850 ? 780 : 640;
                return await SendBatchAsync(imagePath, batch, model, token, smallerSide);
            }

            if (contextOverflow && batch.Count > 1)
            {
                var splitResults = new List<VisionTranslation>();
                foreach (var item in batch)
                    splitResults.AddRange(await SendBatchAsync(imagePath, [item], model, token, 720));
                return splitResults;
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
            throw new InvalidOperationException("Vision LLM 응답이 비어 있습니다.");

        var parsed = ParseResult(content, batchLines);

        if (parsed.Count == 0)
        {
            if (batch.Count > 1)
            {
                var splitResults = new List<VisionTranslation>();
                foreach (var item in batch)
                    splitResults.AddRange(await SendBatchAsync(imagePath, [item], model, token, 720));
                return splitResults;
            }

            throw new InvalidOperationException(
                $"Vision LLM JSON을 해석하지 못했습니다: {Compact(content)}");
        }

        var final = parsed.Select(x =>
        {
            var sourceItem = batch[x.Id];
            return new VisionTranslation(
                sourceItem.GlobalId,
                sourceItem.Line,
                x.CorrectedText,
                x.Translation);
        }).ToList();

        if (parsed.Count < batch.Count)
        {
            var returnedLocalIds = parsed.Select(x => x.Id).ToHashSet();
            for (int localId = 0; localId < batch.Count; localId++)
            {
                if (returnedLocalIds.Contains(localId)) continue;
                final.AddRange(await SendBatchAsync(imagePath, [batch[localId]], model, token, 720));
            }
        }

        return final;
    }

    static CropPayload CreateCropPayload(
        string imagePath,
        IEnumerable<OcrLine> lines,
        int maxImageSide)
    {
        using var input = File.OpenRead(imagePath);
        var frame = BitmapDecoder.Create(
            input,
            BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad).Frames[0];

        var bounds = GetBounds(lines);
        double width = Math.Max(1, bounds.Right - bounds.X);
        double height = Math.Max(1, bounds.Bottom - bounds.Y);

        double padX = Math.Clamp(width * 0.12, 28, 90);
        double padY = Math.Clamp(height * 0.18, 28, 100);

        int x = Math.Max(0, (int)Math.Floor(bounds.X - padX));
        int y = Math.Max(0, (int)Math.Floor(bounds.Y - padY));
        int right = Math.Min(frame.PixelWidth, (int)Math.Ceiling(bounds.Right + padX));
        int bottom = Math.Min(frame.PixelHeight, (int)Math.Ceiling(bounds.Bottom + padY));
        int cropWidth = Math.Max(1, right - x);
        int cropHeight = Math.Max(1, bottom - y);

        BitmapSource source = new CroppedBitmap(frame, new Int32Rect(x, y, cropWidth, cropHeight));

        double longSide = Math.Max(cropWidth, cropHeight);
        double scale = 1.0;

        if (longSide > maxImageSide)
            scale = maxImageSide / longSide;
        else if (longSide < 620)
            scale = Math.Min(1.7, 760.0 / longSide);

        if (Math.Abs(scale - 1.0) > 0.01)
        {
            source = new TransformedBitmap(
                source,
                new ScaleTransform(scale, scale));
        }

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

    static (double X, double Y, double Right, double Bottom) GetBounds(IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();
        if (list.Count == 0)
            return (0, 0, 1, 1);

        return (
            list.Min(x => x.X),
            list.Min(x => x.Y),
            list.Max(x => x.X + x.W),
            list.Max(x => x.Y + x.H));
    }

    static string BuildPrompt(object ocrPayload)
    {
        var json = JsonSerializer.Serialize(ocrPayload);

        const string instructions = """
너는 일본어/영어 만화 이미지 OCR 검수 및 한국어 번역기다.
첨부 이미지는 원본 페이지에서 OCR 영역 주변만 잘라낸 이미지다.
아래 OCR 좌표는 이 잘라낸 이미지 기준이다.

목표:
1. OCR 텍스트가 이미지와 다르면 corrected에 정확한 원문을 적는다.
2. OCR이 맞으면 corrected에 동일한 원문을 적는다.
3. translation에는 자연스러운 한국어 번역만 적는다.
4. 말투, 호칭, 감정, 문장부호를 이미지 문맥에 맞게 유지한다.
5. 항목을 추가/삭제/병합하지 말고 반드시 입력 id를 그대로 유지한다.
6. 추론 과정은 출력하지 말고 즉시 JSON 결과만 반환한다.
7. 설명이나 마크다운 없이 JSON만 출력한다.

출력 형식:
{
  "regions": [
    { "id": 0, "corrected": "원문", "translation": "한국어" }
  ]
}

OCR:
""";

        return instructions + Environment.NewLine + json;
    }

    static List<VisionTranslation> ParseResult(string content, IReadOnlyList<OcrLine> lines)
    {
        try
        {
            var normalized = content.Trim();
            var fenceToken = new string((char)96, 3);
            if (normalized.StartsWith(fenceToken))
            {
                var firstNewLine = normalized.IndexOf('\n');
                if (firstNewLine >= 0) normalized = normalized[(firstNewLine + 1)..];
                var fence = normalized.LastIndexOf(fenceToken, StringComparison.Ordinal);
                if (fence >= 0) normalized = normalized[..fence].Trim();
            }

            using var doc = JsonDocument.Parse(normalized);
            var root = doc.RootElement;
            if (!root.TryGetProperty("regions", out var regions) ||
                regions.ValueKind != JsonValueKind.Array)
                return [];

            var result = new List<VisionTranslation>();
            var used = new HashSet<int>();

            foreach (var region in regions.EnumerateArray())
            {
                if (!region.TryGetProperty("id", out var idEl) ||
                    !idEl.TryGetInt32(out var id))
                    continue;

                if (id < 0 || id >= lines.Count || !used.Add(id))
                    continue;

                var corrected = region.TryGetProperty("corrected", out var correctedEl)
                    ? correctedEl.GetString()?.Trim()
                    : null;

                var translation = region.TryGetProperty("translation", out var translationEl)
                    ? translationEl.GetString()?.Trim()
                    : null;

                result.Add(new VisionTranslation(
                    id,
                    lines[id],
                    string.IsNullOrWhiteSpace(corrected) ? lines[id].Text : corrected,
                    string.IsNullOrWhiteSpace(translation)
                        ? corrected ?? lines[id].Text
                        : translation));
            }

            return result.OrderBy(x => x.Id).ToList();
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
