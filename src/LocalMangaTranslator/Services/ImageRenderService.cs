using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public sealed class ImageRenderService
{
    const double InpaintRadius = 2.5;

    sealed record ContainerLayout(
        OpenCvSharp.Rect Bounds,
        OpenCvSharp.Rect Inner,
        bool DarkBackground,
        bool Detected,
        string Type);

    public async Task RenderAsync(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        string outputPath,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        var renderRegions = regions.Where(x => x.Render).ToList();
        if (renderRegions.Count == 0)
            throw new InvalidOperationException("조판할 번역 영역이 없습니다.");

        string cleanedPath = Path.Combine(
            Path.GetTempPath(),
            $"lmt_inpaint_{Guid.NewGuid():N}.png");

        Dictionary<int, ContainerLayout> layouts;

        try
        {
            progress?.Report($"말풍선/캡션 분석 + 글자 마스크 생성 · {renderRegions.Count}개 블록");

            layouts = await Task.Run(
                () => InpaintPreservingContainers(sourcePath, cleanedPath, renderRegions, token),
                token);

            int detected = layouts.Values.Count(x => x.Detected);
            progress?.Report($"컨테이너 감지 {detected}/{renderRegions.Count} · 글자만 제거 완료");

            token.ThrowIfCancellationRequested();
            progress?.Report("자동 줄바꿈·글자 크기 조절 + 한글 조판 시작");

            TypesetAndSave(cleanedPath, renderRegions, layouts, outputPath, token);

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

    static Dictionary<int, ContainerLayout> InpaintPreservingContainers(
        string sourcePath,
        string cleanedPath,
        IReadOnlyList<VisionTranslation> regions,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidOperationException("원본 이미지를 열 수 없습니다.");

        using var mask = Mat.Zeros(source.Rows, source.Cols, MatType.CV_8UC1).ToMat();
        var layouts = new Dictionary<int, ContainerLayout>(regions.Count);

        foreach (var region in regions)
        {
            token.ThrowIfCancellationRequested();

            var block = region.Source;
            var blockRect = ClampRect(
                (int)Math.Floor(block.X),
                (int)Math.Floor(block.Y),
                (int)Math.Ceiling(block.W),
                (int)Math.Ceiling(block.H),
                source.Cols,
                source.Rows);

            var (containerRect, detected) = DetectContainerRect(
                source,
                blockRect,
                region.Type);

            var innerRect = ComputeInnerRect(
                containerRect,
                blockRect,
                detected,
                region.Type,
                source.Cols,
                source.Rows);

            bool darkBackground = EstimateDarkBackground(
                source,
                innerRect,
                block.Lines);

            layouts[region.Id] = new ContainerLayout(
                containerRect,
                innerRect,
                darkBackground,
                detected,
                region.Type);

            foreach (var line in block.Lines)
            {
                token.ThrowIfCancellationRequested();
                AddTextOnlyMask(
                    source,
                    mask,
                    line,
                    detected ? containerRect : (OpenCvSharp.Rect?)null);
            }
        }

        // 글자 가장자리의 안티앨리어싱까지 최소 범위로 포함한다.
        using (var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new OpenCvSharp.Size(2, 2)))
        {
            Cv2.Dilate(mask, mask, kernel, iterations: 1);
        }

        token.ThrowIfCancellationRequested();

        using var cleaned = new Mat();
        Cv2.Inpaint(source, mask, cleaned, InpaintRadius, InpaintTypes.Telea);

        if (!Cv2.ImWrite(cleanedPath, cleaned))
            throw new InvalidOperationException("인페인트 결과 이미지를 저장하지 못했습니다.");

        return layouts;
    }

    static (OpenCvSharp.Rect Rect, bool Detected) DetectContainerRect(
        Mat source,
        OpenCvSharp.Rect blockRect,
        string type)
    {
        bool caption = string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);

        int padX = (int)Math.Clamp(
            blockRect.Width * (caption ? 0.60 : 0.90),
            32,
            260);

        int padY = (int)Math.Clamp(
            blockRect.Height * (caption ? 0.75 : 1.40),
            28,
            220);

        var search = ClampRect(
            blockRect.X - padX,
            blockRect.Y - padY,
            blockRect.Width + padX * 2,
            blockRect.Height + padY * 2,
            source.Cols,
            source.Rows);

        using var roi = new Mat(source, search);
        using var gray = new Mat();
        using var blurred = new Mat();
        using var edges = new Mat();

        Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(3, 3), 0);
        Cv2.Canny(blurred, edges, 35, 110);

        using (var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new OpenCvSharp.Size(3, 3)))
        {
            Cv2.MorphologyEx(
                edges,
                edges,
                MorphTypes.Close,
                kernel,
                iterations: 1);
        }

        Cv2.FindContours(
            edges,
            out OpenCvSharp.Point[][] contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        double blockArea = Math.Max(1, (double)blockRect.Width * blockRect.Height);
        double blockCenterX = blockRect.X + blockRect.Width / 2.0;
        double blockCenterY = blockRect.Y + blockRect.Height / 2.0;

        OpenCvSharp.Rect? bestRect = null;
        double bestScore = double.NegativeInfinity;

        foreach (var contour in contours)
        {
            if (contour.Length < 4)
                continue;

            var local = Cv2.BoundingRect(contour);
            var candidate = new OpenCvSharp.Rect(
                search.X + local.X,
                search.Y + local.Y,
                local.Width,
                local.Height);

            if (candidate.Width < Math.Max(12, blockRect.Width * 0.90) ||
                candidate.Height < Math.Max(10, blockRect.Height * 0.90))
                continue;

            double coverage = IntersectionArea(candidate, blockRect) / blockArea;
            if (coverage < 0.65)
                continue;

            if (!Contains(candidate, blockCenterX, blockCenterY))
                continue;

            double ratio =
                Math.Max(1, (double)candidate.Width * candidate.Height) /
                blockArea;

            double maxRatio = caption ? 14.0 : 20.0;
            if (ratio < 1.08 || ratio > maxRatio)
                continue;

            // 검색영역 자체나 패널 테두리를 잘못 잡는 경우를 억제한다.
            int touches = 0;
            const int edgeTolerance = 3;
            if (candidate.Left <= search.Left + edgeTolerance) touches++;
            if (candidate.Top <= search.Top + edgeTolerance) touches++;
            if (candidate.Right >= search.Right - edgeTolerance) touches++;
            if (candidate.Bottom >= search.Bottom - edgeTolerance) touches++;
            if (touches >= 3)
                continue;

            double centerPenalty =
                Math.Abs((candidate.X + candidate.Width / 2.0) - blockCenterX) /
                Math.Max(1, candidate.Width) +
                Math.Abs((candidate.Y + candidate.Height / 2.0) - blockCenterY) /
                Math.Max(1, candidate.Height);

            double preferredRatio = caption ? 1.8 : 2.5;
            double sizePenalty = Math.Abs(Math.Log(ratio / preferredRatio));

            double contourArea = Math.Abs(Cv2.ContourArea(contour));
            double rectangularArea = Math.Max(1, (double)local.Width * local.Height);
            double fillRatio = contourArea / rectangularArea;

            // 말풍선은 불규칙해도 되고, 캡션 박스는 조금 더 꽉 찬 contour를 선호한다.
            double shapeBonus = caption
                ? Math.Clamp(fillRatio, 0, 1) * 0.45
                : Math.Clamp(fillRatio, 0, 1) * 0.15;

            double score =
                coverage * 5.0 -
                sizePenalty -
                centerPenalty +
                shapeBonus;

            if (score > bestScore)
            {
                bestScore = score;
                bestRect = candidate;
            }
        }

        if (bestRect is not null && bestScore >= 1.8)
            return (ClampRect(
                bestRect.Value.X,
                bestRect.Value.Y,
                bestRect.Value.Width,
                bestRect.Value.Height,
                source.Cols,
                source.Rows), true);

        // 컨테이너를 못 찾은 경우에만 기존 OCR 위치를 안전하게 확장해 사용한다.
        double fallbackX = caption ? 0.10 : 0.22;
        double fallbackY = caption ? 0.12 : 0.28;

        int fx = (int)Math.Ceiling(blockRect.Width * fallbackX);
        int fy = (int)Math.Ceiling(blockRect.Height * fallbackY);

        return (ClampRect(
            blockRect.X - fx,
            blockRect.Y - fy,
            blockRect.Width + fx * 2,
            blockRect.Height + fy * 2,
            source.Cols,
            source.Rows), false);
    }

    static OpenCvSharp.Rect ComputeInnerRect(
        OpenCvSharp.Rect container,
        OpenCvSharp.Rect block,
        bool detected,
        string type,
        int imageWidth,
        int imageHeight)
    {
        if (!detected)
        {
            // 말풍선 경계를 못 찾았다면 OCR 블록 확장영역을 그대로 최대한 활용한다.
            return ClampRect(
                container.X,
                container.Y,
                container.Width,
                container.Height,
                imageWidth,
                imageHeight);
        }

        bool caption = string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);

        int padX = (int)Math.Clamp(
            container.Width * (caption ? 0.045 : 0.065),
            5,
            26);

        int padY = (int)Math.Clamp(
            container.Height * (caption ? 0.055 : 0.080),
            4,
            26);

        var inner = ClampRect(
            container.X + padX,
            container.Y + padY,
            Math.Max(12, container.Width - padX * 2),
            Math.Max(12, container.Height - padY * 2),
            imageWidth,
            imageHeight);

        // 너무 작은 contour를 잡았을 경우 원래 OCR 텍스트보다 조판영역이 작아지는 것을 방지한다.
        if (inner.Width < block.Width * 0.82 || inner.Height < block.Height * 0.78)
        {
            int bx = Math.Max(4, (int)Math.Ceiling(block.Width * 0.06));
            int by = Math.Max(4, (int)Math.Ceiling(block.Height * 0.08));

            inner = ClampRect(
                Math.Min(inner.X, block.X - bx),
                Math.Min(inner.Y, block.Y - by),
                Math.Max(inner.Right, block.Right + bx) - Math.Min(inner.X, block.X - bx),
                Math.Max(inner.Bottom, block.Bottom + by) - Math.Min(inner.Y, block.Y - by),
                imageWidth,
                imageHeight);
        }

        return inner;
    }

    static void AddTextOnlyMask(
        Mat source,
        Mat globalMask,
        OcrLine line,
        OpenCvSharp.Rect? container)
    {
        int expand = Math.Max(1, (int)Math.Ceiling(line.H * 0.035));

        var textRect = ClampRect(
            (int)Math.Floor(line.X) - expand,
            (int)Math.Floor(line.Y) - expand,
            (int)Math.Ceiling(line.W) + expand * 2,
            (int)Math.Ceiling(line.H) + expand * 2,
            source.Cols,
            source.Rows);

        if (container is not null)
        {
            var clipped = Intersect(textRect, container.Value);
            if (clipped.Width <= 0 || clipped.Height <= 0)
                return;
            textRect = clipped;
        }

        int ringPad = (int)Math.Clamp(line.H * 0.35, 4, 16);
        var sampleRect = ClampRect(
            textRect.X - ringPad,
            textRect.Y - ringPad,
            textRect.Width + ringPad * 2,
            textRect.Height + ringPad * 2,
            source.Cols,
            source.Rows);

        if (container is not null)
        {
            var clipped = Intersect(sampleRect, container.Value);
            if (clipped.Width > 0 && clipped.Height > 0)
                sampleRect = clipped;
        }

        var samples = CollectRingSamples(source, sampleRect, textRect);
        if (samples.Count < 12)
        {
            using var roi = new Mat(source, sampleRect);
            var mean = Cv2.Mean(roi);
            samples.Add(new Vec3b(
                (byte)Math.Clamp((int)Math.Round(mean.Val0), 0, 255),
                (byte)Math.Clamp((int)Math.Round(mean.Val1), 0, 255),
                (byte)Math.Clamp((int)Math.Round(mean.Val2), 0, 255)));
        }

        var background = MedianColor(samples);

        var ringDistances = samples
            .Select(p => ColorDistance(p, background))
            .OrderBy(x => x)
            .ToList();

        double naturalVariation = Percentile(ringDistances, 0.82);
        double threshold = Math.Clamp(naturalVariation + 18, 28, 78);

        int marked = 0;
        int total = Math.Max(1, textRect.Width * textRect.Height);

        for (int y = textRect.Top; y < textRect.Bottom; y++)
        {
            for (int x = textRect.Left; x < textRect.Right; x++)
            {
                var pixel = source.At<Vec3b>(y, x);
                double distance = ColorDistance(pixel, background);

                double bgLuma = Luminance(background);
                double pxLuma = Luminance(pixel);
                double lumaDelta = Math.Abs(pxLuma - bgLuma);

                if (distance >= threshold || lumaDelta >= Math.Max(24, threshold * 0.72))
                {
                    globalMask.Set(y, x, (byte)255);
                    marked++;
                }
            }
        }

        // 배경 변화가 큰 영역에서 보수적 임계값 때문에 글자를 거의 못 잡은 경우만
        // 밝기 대비를 이용해 한 번 더 좁게 보강한다.
        if (marked / (double)total < 0.012)
        {
            double bgLuma = Luminance(background);

            for (int y = textRect.Top; y < textRect.Bottom; y++)
            {
                for (int x = textRect.Left; x < textRect.Right; x++)
                {
                    var pixel = source.At<Vec3b>(y, x);
                    double delta = Luminance(pixel) - bgLuma;

                    bool likelyText = bgLuma >= 145
                        ? delta <= -32
                        : bgLuma <= 110
                            ? delta >= 32
                            : Math.Abs(delta) >= 38;

                    if (likelyText)
                        globalMask.Set(y, x, (byte)255);
                }
            }
        }
    }

    static List<Vec3b> CollectRingSamples(
        Mat source,
        OpenCvSharp.Rect outer,
        OpenCvSharp.Rect inner)
    {
        var samples = new List<Vec3b>();
        int step = outer.Width * outer.Height > 18000 ? 3 : 2;

        for (int y = outer.Top; y < outer.Bottom; y += step)
        {
            for (int x = outer.Left; x < outer.Right; x += step)
            {
                if (x >= inner.Left && x < inner.Right &&
                    y >= inner.Top && y < inner.Bottom)
                    continue;

                samples.Add(source.At<Vec3b>(y, x));
            }
        }

        return samples;
    }

    static Vec3b MedianColor(IReadOnlyList<Vec3b> samples)
    {
        if (samples.Count == 0)
            return new Vec3b(255, 255, 255);

        var b = samples.Select(x => (int)x.Item0).OrderBy(x => x).ToArray();
        var g = samples.Select(x => (int)x.Item1).OrderBy(x => x).ToArray();
        var r = samples.Select(x => (int)x.Item2).OrderBy(x => x).ToArray();

        int mid = samples.Count / 2;
        return new Vec3b(
            (byte)b[mid],
            (byte)g[mid],
            (byte)r[mid]);
    }

    static bool EstimateDarkBackground(
        Mat source,
        OpenCvSharp.Rect inner,
        IReadOnlyList<OcrLine> lines)
    {
        var luminances = new List<double>();
        int step = inner.Width * inner.Height > 30000 ? 4 : 3;

        for (int y = inner.Top; y < inner.Bottom; y += step)
        {
            for (int x = inner.Left; x < inner.Right; x += step)
            {
                bool insideText = lines.Any(line =>
                    x >= line.X - 2 &&
                    x <= line.X + line.W + 2 &&
                    y >= line.Y - 2 &&
                    y <= line.Y + line.H + 2);

                if (insideText)
                    continue;

                luminances.Add(Luminance(source.At<Vec3b>(y, x)));
            }
        }

        if (luminances.Count == 0)
        {
            using var roi = new Mat(source, inner);
            var mean = Cv2.Mean(roi);
            double fallback = mean.Val2 * 0.299 + mean.Val1 * 0.587 + mean.Val0 * 0.114;
            return fallback < 125;
        }

        luminances.Sort();
        double median = luminances[luminances.Count / 2];
        return median < 125;
    }

    static void TypesetAndSave(
        string cleanedPath,
        IReadOnlyList<VisionTranslation> regions,
        IReadOnlyDictionary<int, ContainerLayout> layouts,
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

                string text = NormalizeForAutoLayout(region.Translation);
                if (string.IsNullOrWhiteSpace(text))
                    continue;

                ContainerLayout layout;
                if (!layouts.TryGetValue(region.Id, out layout!))
                {
                    var fallback = ClampRect(
                        (int)region.Source.X,
                        (int)region.Source.Y,
                        (int)region.Source.W,
                        (int)region.Source.H,
                        width,
                        height);

                    layout = new ContainerLayout(
                        fallback,
                        fallback,
                        false,
                        false,
                        region.Type);
                }

                var box = new System.Windows.Rect(
                    layout.Inner.X,
                    layout.Inner.Y,
                    Math.Max(12, layout.Inner.Width),
                    Math.Max(12, layout.Inner.Height));

                DrawFittedText(
                    dc,
                    text,
                    box,
                    layout.DarkBackground,
                    region.Type);
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

    static string NormalizeForAutoLayout(string text)
    {
        var normalized = text
            .Replace("[BR]", " ", StringComparison.OrdinalIgnoreCase)
            .Replace('\r', ' ')
            .Replace('\n', ' ')
            .Trim();

        return string.Join(
            " ",
            normalized.Split(
                (char[]?)null,
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
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

    static void DrawFittedText(
        DrawingContext dc,
        string text,
        System.Windows.Rect box,
        bool darkBackground,
        string type)
    {
        var culture = CultureInfo.GetCultureInfo("ko-KR");
        var typeface = new Typeface(
            new FontFamily("Malgun Gothic"),
            FontStyles.Normal,
            FontWeights.SemiBold,
            FontStretches.Normal);

        bool caption = string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);
        var alignment = caption ? TextAlignment.Left : TextAlignment.Center;

        double minFont = Math.Clamp(
            Math.Min(box.Width, box.Height) * 0.055,
            8,
            13);

        double maxFont = Math.Clamp(
            caption
                ? Math.Min(box.Height * 0.34, 48)
                : Math.Min(box.Height * 0.46, 56),
            Math.Max(minFont, 12),
            caption ? 48 : 56);

        double[] lineHeightFactors = [1.12, 1.06, 1.00];

        FormattedText? best = null;
        double bestSize = 0;
        double bestFactor = 1.12;

        foreach (double factor in lineHeightFactors)
        {
            double low = minFont;
            double high = maxFont;
            FormattedText? localBest = null;
            double localSize = minFont;

            for (int i = 0; i < 13; i++)
            {
                double size = (low + high) / 2.0;
                var candidate = MakeFormattedText(
                    text,
                    culture,
                    typeface,
                    size,
                    box.Width,
                    factor,
                    alignment);

                if (candidate.Height <= box.Height * 0.97)
                {
                    localBest = candidate;
                    localSize = size;
                    low = size;
                }
                else
                {
                    high = size;
                }
            }

            localBest ??= MakeFormattedText(
                text,
                culture,
                typeface,
                minFont,
                box.Width,
                factor,
                alignment);

            // 줄간격을 줄이는 대신 눈에 띄게 더 큰 글자를 쓸 수 있을 때만 채택한다.
            double readabilityScore =
                localSize -
                (1.12 - factor) * 12.0;

            double currentScore =
                bestSize -
                (1.12 - bestFactor) * 12.0;

            if (best is null || readabilityScore > currentScore)
            {
                best = localBest;
                bestSize = localSize;
                bestFactor = factor;
            }
        }

        best ??= MakeFormattedText(
            text,
            culture,
            typeface,
            minFont,
            box.Width,
            1.0,
            alignment);

        // 최소 글자 크기에서도 높이가 넘치면 더 작게 내려서 박스 밖으로는 절대 나가지 않게 한다.
        if (best.Height > box.Height * 0.98)
        {
            double emergencySize = minFont;
            while (emergencySize > 6.0)
            {
                emergencySize -= 0.5;
                var candidate = MakeFormattedText(
                    text,
                    culture,
                    typeface,
                    emergencySize,
                    box.Width,
                    1.0,
                    alignment);

                best = candidate;
                bestSize = emergencySize;

                if (candidate.Height <= box.Height * 0.98)
                    break;
            }
        }

        double originY = caption
            ? box.Y
            : box.Y + Math.Max(0, (box.Height - best.Height) / 2.0);

        var origin = new System.Windows.Point(box.X, originY);

        Brush fill = darkBackground ? Brushes.White : Brushes.Black;
        Brush outline = darkBackground ? Brushes.Black : Brushes.White;

        var geometry = best.BuildGeometry(origin);
        var pen = new Pen(
            outline,
            Math.Clamp(bestSize * 0.035, 0.45, 1.35));

        dc.DrawGeometry(fill, pen, geometry);
    }

    static FormattedText MakeFormattedText(
        string text,
        CultureInfo culture,
        Typeface typeface,
        double fontSize,
        double maxWidth,
        double lineHeightFactor,
        TextAlignment alignment)
    {
        return new FormattedText(
            text,
            culture,
            FlowDirection.LeftToRight,
            typeface,
            fontSize,
            Brushes.Black,
            1.0)
        {
            TextAlignment = alignment,
            MaxTextWidth = Math.Max(12, maxWidth),
            LineHeight = fontSize * lineHeightFactor,
            Trimming = TextTrimming.None
        };
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
        int right = Math.Clamp(x + Math.Max(1, width), left + 1, imageWidth);
        int bottom = Math.Clamp(y + Math.Max(1, height), top + 1, imageHeight);

        return new OpenCvSharp.Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }

    static OpenCvSharp.Rect Intersect(
        OpenCvSharp.Rect a,
        OpenCvSharp.Rect b)
    {
        int left = Math.Max(a.Left, b.Left);
        int top = Math.Max(a.Top, b.Top);
        int right = Math.Min(a.Right, b.Right);
        int bottom = Math.Min(a.Bottom, b.Bottom);

        if (right <= left || bottom <= top)
            return new OpenCvSharp.Rect(0, 0, 0, 0);

        return new OpenCvSharp.Rect(
            left,
            top,
            right - left,
            bottom - top);
    }

    static double IntersectionArea(
        OpenCvSharp.Rect a,
        OpenCvSharp.Rect b)
    {
        var intersection = Intersect(a, b);
        return Math.Max(0, (double)intersection.Width * intersection.Height);
    }

    static bool Contains(
        OpenCvSharp.Rect rect,
        double x,
        double y)
        => x >= rect.Left &&
           x <= rect.Right &&
           y >= rect.Top &&
           y <= rect.Bottom;

    static double ColorDistance(Vec3b a, Vec3b b)
    {
        double db = a.Item0 - b.Item0;
        double dg = a.Item1 - b.Item1;
        double dr = a.Item2 - b.Item2;
        return Math.Sqrt(db * db + dg * dg + dr * dr);
    }

    static double Luminance(Vec3b p)
        => p.Item2 * 0.299 +
           p.Item1 * 0.587 +
           p.Item0 * 0.114;

    static double Percentile(IReadOnlyList<double> sorted, double percentile)
    {
        if (sorted.Count == 0)
            return 0;

        double p = Math.Clamp(percentile, 0, 1);
        int index = (int)Math.Round((sorted.Count - 1) * p);
        return sorted[Math.Clamp(index, 0, sorted.Count - 1)];
    }
}
