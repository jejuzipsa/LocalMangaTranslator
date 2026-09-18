using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public sealed class ImageRenderService
{
    const double InpaintRadius = 3.0;

    public async Task RenderAsync(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        string outputPath,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (regions.Count == 0)
            throw new InvalidOperationException("조판할 번역 영역이 없습니다.");

        string cleanedPath = Path.Combine(
            Path.GetTempPath(),
            $"lmt_inpaint_{Guid.NewGuid():N}.png");

        Dictionary<int, bool> darkBackground;

        try
        {
            progress?.Report($"인페인트 마스크 생성 · {regions.Count}개 영역");

            darkBackground = await Task.Run(
                () => Inpaint(sourcePath, cleanedPath, regions, token),
                token);

            token.ThrowIfCancellationRequested();
            progress?.Report("인페인트 완료 · 한글 조판 시작");

            TypesetAndSave(cleanedPath, regions, darkBackground, outputPath, token);

            progress?.Report($"완성 이미지 저장 · {Path.GetFileName(outputPath)}");
        }
        finally
        {
            try
            {
                if (File.Exists(cleanedPath))
                    File.Delete(cleanedPath);
            }
            catch { }
        }
    }

    static Dictionary<int, bool> Inpaint(
        string sourcePath,
        string cleanedPath,
        IReadOnlyList<VisionTranslation> regions,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidOperationException("원본 이미지를 열 수 없습니다.");

        using var mask = Mat.Zeros(source.Rows, source.Cols, MatType.CV_8UC1).ToMat();
        var darkBackground = new Dictionary<int, bool>(regions.Count);

        foreach (var region in regions)
        {
            token.ThrowIfCancellationRequested();

            var line = region.Source;
            int padX = Math.Max(4, (int)Math.Ceiling(line.W * 0.10));
            int padY = Math.Max(4, (int)Math.Ceiling(line.H * 0.14));

            var rect = ClampRect(
                (int)Math.Floor(line.X) - padX,
                (int)Math.Floor(line.Y) - padY,
                (int)Math.Ceiling(line.W) + padX * 2,
                (int)Math.Ceiling(line.H) + padY * 2,
                source.Cols,
                source.Rows);

            if (rect.Width <= 1 || rect.Height <= 1)
                continue;

            using (var roi = new Mat(source, rect))
            {
                var mean = Cv2.Mean(roi);
                double luminance = mean.Val2 * 0.299 + mean.Val1 * 0.587 + mean.Val0 * 0.114;
                darkBackground[region.Id] = luminance < 125;
            }

            Cv2.Rectangle(mask, rect, Scalar.White, -1);
        }

        using (var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new OpenCvSharp.Size(3, 3)))
        {
            Cv2.Dilate(mask, mask, kernel, iterations: 1);
        }

        token.ThrowIfCancellationRequested();

        using var cleaned = new Mat();
        Cv2.Inpaint(source, mask, cleaned, InpaintRadius, InpaintMethod.Telea);

        if (!Cv2.ImWrite(cleanedPath, cleaned))
            throw new InvalidOperationException("인페인트 결과 이미지를 저장하지 못했습니다.");

        return darkBackground;
    }

    static OpenCvSharp.Rect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left = Math.Clamp(x, 0, Math.Max(0, imageWidth - 1));
        int top = Math.Clamp(y, 0, Math.Max(0, imageHeight - 1));
        int right = Math.Clamp(x + width, left + 1, imageWidth);
        int bottom = Math.Clamp(y + height, top + 1, imageHeight);

        return new OpenCvSharp.Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }

    static void TypesetAndSave(
        string cleanedPath,
        IReadOnlyList<VisionTranslation> regions,
        IReadOnlyDictionary<int, bool> darkBackground,
        string outputPath,
        CancellationToken token)
    {
        var bitmap = LoadBitmap(cleanedPath);
        int width = bitmap.PixelWidth;
        int height = bitmap.PixelHeight;

        var visual = new DrawingVisual();

        using (var dc = visual.RenderOpen())
        {
            dc.DrawImage(bitmap, new System.Windows.Rect(0, 0, width, height));

            foreach (var region in regions.OrderBy(x => x.Id))
            {
                token.ThrowIfCancellationRequested();

                string text = region.Translation.Trim();
                if (string.IsNullOrWhiteSpace(text))
                    continue;

                var box = CreateTextBox(region.Source, width, height);
                bool dark = darkBackground.TryGetValue(region.Id, out var value) && value;

                DrawFittedText(dc, text, box, dark);
            }
        }

        var rendered = new RenderTargetBitmap(
            width,
            height,
            96,
            96,
            PixelFormats.Pbgra32);

        rendered.Render(visual);
        rendered.Freeze();

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(rendered));

        using var stream = File.Create(outputPath);
        encoder.Save(stream);
    }

    static BitmapSource LoadBitmap(string path)
    {
        using var input = File.OpenRead(path);

        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = input;
        bitmap.EndInit();
        bitmap.Freeze();

        return bitmap;
    }

    static System.Windows.Rect CreateTextBox(OcrLine line, int imageWidth, int imageHeight)
    {
        bool vertical = line.H > line.W * 1.35;

        double targetWidth = vertical
            ? Math.Max(line.W * 2.8, line.H * 0.72)
            : line.W * 1.22;

        double targetHeight = vertical
            ? line.H * 1.05
            : line.H * 1.55;

        targetWidth = Math.Max(targetWidth, line.W + 12);
        targetHeight = Math.Max(targetHeight, line.H + 10);

        double centerX = line.X + line.W / 2;
        double centerY = line.Y + line.H / 2;

        double x = Math.Clamp(
            centerX - targetWidth / 2,
            0,
            Math.Max(0, imageWidth - targetWidth));

        double y = Math.Clamp(
            centerY - targetHeight / 2,
            0,
            Math.Max(0, imageHeight - targetHeight));

        double width = Math.Min(targetWidth, imageWidth - x);
        double height = Math.Min(targetHeight, imageHeight - y);

        return new System.Windows.Rect(
            x,
            y,
            Math.Max(12, width),
            Math.Max(12, height));
    }

    static void DrawFittedText(
        DrawingContext dc,
        string text,
        System.Windows.Rect box,
        bool darkBackground)
    {
        var culture = CultureInfo.GetCultureInfo("ko-KR");
        var typeface = new Typeface(
            new FontFamily("Malgun Gothic"),
            FontStyles.Normal,
            FontWeights.SemiBold,
            FontStretches.Normal);

        double maxFont = Math.Clamp(
            Math.Min(box.Height * 0.72, 48),
            12,
            48);

        double minFont = Math.Min(9, maxFont);
        double low = minFont;
        double high = maxFont;
        FormattedText? best = null;

        for (int i = 0; i < 10; i++)
        {
            double size = (low + high) / 2;
            var candidate = MakeFormattedText(text, culture, typeface, size, box.Width);

            if (candidate.Height <= box.Height * 0.96)
            {
                best = candidate;
                low = size;
            }
            else
            {
                high = size;
            }
        }

        best ??= MakeFormattedText(text, culture, typeface, minFont, box.Width);

        double originY = box.Y + Math.Max(0, (box.Height - best.Height) / 2);
        var origin = new System.Windows.Point(box.X, originY);

        Brush fill = darkBackground ? Brushes.White : Brushes.Black;
        Brush outline = darkBackground ? Brushes.Black : Brushes.White;

        var geometry = best.BuildGeometry(origin);
        var pen = new Pen(outline, Math.Clamp(best.FontSize * 0.065, 0.8, 2.2));

        dc.DrawGeometry(fill, pen, geometry);
    }

    static FormattedText MakeFormattedText(
        string text,
        CultureInfo culture,
        Typeface typeface,
        double fontSize,
        double maxWidth)
    {
        var formatted = new FormattedText(
            text,
            culture,
            FlowDirection.LeftToRight,
            typeface,
            fontSize,
            Brushes.Black,
            1.0)
        {
            TextAlignment = TextAlignment.Center,
            MaxTextWidth = Math.Max(12, maxWidth),
            LineHeight = fontSize * 1.12,
            Trimming = TextTrimming.None
        };

        return formatted;
    }
}
