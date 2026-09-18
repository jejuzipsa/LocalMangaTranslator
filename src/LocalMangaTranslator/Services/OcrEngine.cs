using System.Security.Cryptography;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using RapidOcrNet;

namespace LocalMangaTranslator.Services;

public sealed record OcrLine(double X, double Y, double W, double H, string Text, float Confidence, string Language);

public sealed class OcrEngine : IDisposable
{
    const string CacheVersion = "v5-modes-sfx";
    const double SecondPassScale = 2.0;
    const double FocusScale = 3.0;

    readonly RapidOcr engine = new();
    readonly string cacheRoot;
    readonly RapidOcrOptions options = RapidOcrOptions.Default with
    {
        TextScore = 0.30f,
        BoxScoreThresh = 0.30f,
        ReturnWordBox = true,
        RecMaxDegreeOfParallelism = 2
    };

    public bool Ready { get; private set; }
    public string Status { get; private set; } = "OCR 초기화 전";

    public OcrEngine(string modelRoot)
    {
        cacheRoot = Path.Combine(AppContext.BaseDirectory, "cache", "ocr");
        Directory.CreateDirectory(cacheRoot);

        try
        {
            var v5 = Path.Combine(modelRoot, "v5");
            var det = Path.Combine(v5, "ch_PP-OCRv5_mobile_det.onnx");
            var cls = Path.Combine(v5, "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx");
            var rec = Path.Combine(v5, "ch_PP-OCRv5_rec_mobile.onnx");
            var keys = Path.Combine(v5, "ppocrv5_ch_dict.txt");

            if (File.Exists(det) && File.Exists(cls) && File.Exists(rec) && File.Exists(keys))
            {
                engine.InitModels(det, cls, rec, keys);
                Status = "PP-OCRv5 중/영/일";
            }
            else
            {
                var oldCwd = Environment.CurrentDirectory;
                try
                {
                    Environment.CurrentDirectory = AppContext.BaseDirectory;
                    engine.InitModels();
                }
                finally
                {
                    Environment.CurrentDirectory = oldCwd;
                }
                Status = "RapidOCR 기본 모델";
            }
            Ready = true;
        }
        catch (Exception ex)
        {
            Ready = false;
            Status = $"OCR 초기화 실패: {ex.Message}";
        }
    }

    public Task<List<OcrLine>> RecognizeAsync(string imagePath, CancellationToken token)
        => Task.Run(() => RecognizeCached(imagePath, token), token);

    List<OcrLine> RecognizeCached(string imagePath, CancellationToken token)
    {
        if (!Ready) return [];

        var cachePath = GetCachePath(imagePath);
        try
        {
            if (File.Exists(cachePath))
            {
                var cached = JsonSerializer.Deserialize<List<OcrLine>>(File.ReadAllText(cachePath));
                if (cached is not null) return cached;
            }
        }
        catch { }

        token.ThrowIfCancellationRequested();
        var lines = RecognizeMultiPass(imagePath, token);
        try { File.WriteAllText(cachePath, JsonSerializer.Serialize(lines)); } catch { }
        return lines;
    }

    List<OcrLine> RecognizeMultiPass(string imagePath, CancellationToken token)
    {
        var merged = DetectToLines(imagePath, 1.0);
        token.ThrowIfCancellationRequested();

        string fullTemp = Path.Combine(Path.GetTempPath(), $"lmt_full_{Guid.NewGuid():N}.png");
        try
        {
            CreateUpscaledImage(imagePath, fullTemp, SecondPassScale);
            MergeLines(merged, DetectToLines(fullTemp, SecondPassScale));
        }
        finally { TryDelete(fullTemp); }

        var anchors = merged.Where(IsDialogueSized).OrderByDescending(x => x.Confidence).Take(40).ToList();
        foreach (var anchor in anchors)
        {
            token.ThrowIfCancellationRequested();
            string cropTemp = Path.Combine(Path.GetTempPath(), $"lmt_focus_{Guid.NewGuid():N}.png");
            try
            {
                var crop = CreateFocusCrop(imagePath, cropTemp, anchor, FocusScale);
                if (crop is null) continue;
                var global = DetectToLines(cropTemp, FocusScale)
                    .Select(x => x with { X = x.X + crop.Value.X, Y = x.Y + crop.Value.Y });
                MergeLines(merged, global);
            }
            catch { }
            finally { TryDelete(cropTemp); }
        }

        return merged.OrderBy(x => x.Y).ThenBy(x => x.X).ToList();
    }

    List<OcrLine> DetectToLines(string imagePath, double coordinateScale)
    {
        var result = engine.Detect(imagePath, options);
        var lines = new List<OcrLine>(result.TextBlocks.Length);

        foreach (var block in result.TextBlocks)
        {
            if (string.IsNullOrWhiteSpace(block.Text) || block.BoxPoints.Length == 0) continue;

            double minX = block.BoxPoints.Min(p => p.X) / coordinateScale;
            double minY = block.BoxPoints.Min(p => p.Y) / coordinateScale;
            double maxX = block.BoxPoints.Max(p => p.X) / coordinateScale;
            double maxY = block.BoxPoints.Max(p => p.Y) / coordinateScale;
            float confidence = block.CharScores is { Length: > 0 }
                ? block.CharScores.Average()
                : block.BoxScore;

            lines.Add(new OcrLine(
                minX, minY,
                Math.Max(1, maxX - minX),
                Math.Max(1, maxY - minY),
                block.Text.Trim(),
                confidence,
                GuessLanguage(block.Text)));
        }

        return lines;
    }

    static bool IsDialogueSized(OcrLine x)
    {
        double longSide = Math.Max(x.W, x.H);
        double shortSide = Math.Min(x.W, x.H);
        return x.Confidence >= 0.45f && longSide >= 12 && shortSide >= 4 && x.W * x.H < 90000;
    }

    static (double X, double Y)? CreateFocusCrop(string source, string destination, OcrLine anchor, double scale)
    {
        using var input = File.OpenRead(source);
        var frame = BitmapDecoder.Create(input, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];

        double padX = Math.Max(55, anchor.W * 1.8);
        double padY = Math.Max(55, anchor.H * 2.2);
        int x = Math.Max(0, (int)Math.Floor(anchor.X - padX));
        int y = Math.Max(0, (int)Math.Floor(anchor.Y - padY));
        int right = Math.Min(frame.PixelWidth, (int)Math.Ceiling(anchor.X + anchor.W + padX));
        int bottom = Math.Min(frame.PixelHeight, (int)Math.Ceiling(anchor.Y + anchor.H + padY));
        int w = right - x;
        int h = bottom - y;
        if (w < 24 || h < 24) return null;

        var transformed = new TransformedBitmap(
            new CroppedBitmap(frame, new Int32Rect(x, y, w, h)),
            new ScaleTransform(scale, scale));

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(transformed));
        using var output = File.Create(destination);
        encoder.Save(output);
        return (x, y);
    }

    static void CreateUpscaledImage(string source, string destination, double scale)
    {
        using var input = File.OpenRead(source);
        var frame = BitmapDecoder.Create(input, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
        var transformed = new TransformedBitmap(frame, new ScaleTransform(scale, scale));
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(transformed));
        using var output = File.Create(destination);
        encoder.Save(output);
    }

    static void MergeLines(List<OcrLine> target, IEnumerable<OcrLine> candidates)
    {
        foreach (var candidate in candidates)
        {
            int i = target.FindIndex(existing => IsDuplicate(existing, candidate));
            if (i < 0) target.Add(candidate);
            else if (candidate.Confidence > target[i].Confidence) target[i] = candidate;
        }
    }

    static bool IsDuplicate(OcrLine a, OcrLine b)
    {
        double left = Math.Max(a.X, b.X);
        double top = Math.Max(a.Y, b.Y);
        double right = Math.Min(a.X + a.W, b.X + b.W);
        double bottom = Math.Min(a.Y + a.H, b.Y + b.H);
        double intersection = Math.Max(0, right - left) * Math.Max(0, bottom - top);
        double minArea = Math.Min(a.W * a.H, b.W * b.H);

        if (minArea > 0 && intersection / minArea >= 0.55) return true;

        double dx = a.X + a.W / 2 - (b.X + b.W / 2);
        double dy = a.Y + a.H / 2 - (b.Y + b.H / 2);
        double distance = Math.Sqrt(dx * dx + dy * dy);
        return distance < Math.Max(8, Math.Min(Math.Max(a.W, a.H), Math.Max(b.W, b.H)) * .35)
            && string.Equals(Normalize(a.Text), Normalize(b.Text), StringComparison.OrdinalIgnoreCase);
    }

    static string Normalize(string text) => new(text.Where(char.IsLetterOrDigit).ToArray());

    static string GuessLanguage(string text)
    {
        if (text.Any(c => c is >= '\u3040' and <= '\u30ff')) return "ja";
        if (text.Any(c => c is >= '\uac00' and <= '\ud7a3')) return "ko";
        if (text.Any(c => c is >= '\u3400' and <= '\u9fff')) return "cjk";
        return "en";
    }

    string GetCachePath(string imagePath)
    {
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(imagePath);
        return Path.Combine(cacheRoot, $"{CacheVersion}_{Convert.ToHexString(sha.ComputeHash(stream))}.json");
    }

    static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { }
    }

    public void Dispose() => engine.Dispose();
}
