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

    public async Task RenderAsync(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        string outputPath,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        var requestedRegions = regions.Where(x => x.Render).ToList();
        var renderRegions = SuppressDuplicateRegions(
            requestedRegions,
            out var suppressedDuplicates);

        if (renderRegions.Count == 0)
            throw new InvalidOperationException("조판할 번역 영역이 없습니다.");

        if (suppressedDuplicates.Count > 0)
        {
            progress?.Report(
                $"중복 OCR/번역 영역 {suppressedDuplicates.Count}개 제외 · " +
                string.Join(", ", suppressedDuplicates.Select(x => $"id={x.Id}")));
        }

        string cleanedPath = Path.Combine(
            Path.GetTempPath(),
            $"lmt_inpaint_{Guid.NewGuid():N}.png");

        Dictionary<int, BalloonLayout> layouts;

        var debugDir = Path.Combine(
            Path.GetDirectoryName(outputPath) ?? AppContext.BaseDirectory,
            "debug");

        var debugPath = Path.Combine(
            debugDir,
            Path.GetFileNameWithoutExtension(outputPath) + ".balloon-debug.png");

        try
        {
            progress?.Report($"말풍선 마스크 분석 + 글자 마스크 생성 · {renderRegions.Count}개 블록");

            layouts = await Task.Run(
                () => InpaintPreservingContainers(
                    sourcePath,
                    cleanedPath,
                    debugPath,
                    renderRegions,
                    progress,
                    token),
                token);

            int detected = layouts.Values.Count(x => x.Detected && x.ShouldRender);
            int fallback = layouts.Values.Count(x => x.Mode == "fallback" && x.ShouldRender);
            int rejected = layouts.Values.Count(x => !x.ShouldRender);

            progress?.Report(
                $"말풍선 마스크 성공 {detected} · fallback {fallback} · 원문 유지 {rejected}");

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

    static Dictionary<int, BalloonLayout> InpaintPreservingContainers(
        string sourcePath,
        string cleanedPath,
        string debugPath,
        IReadOnlyList<VisionTranslation> regions,
        IProgress<string>? progress,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidOperationException("원본 이미지를 열 수 없습니다.");

        using var mask = Mat.Zeros(source.Rows, source.Cols, MatType.CV_8UC1).ToMat();
        var layouts = new Dictionary<int, BalloonLayout>(regions.Count);
        int detachedEraseCount = 0;
        int detachedKeepCount = 0;

        foreach (var region in regions)
        {
            token.ThrowIfCancellationRequested();

            var block = region.Source;
            var layout = BalloonMaskService.Analyze(
                source,
                block,
                region.Type);

            layouts[region.Id] = layout;
            progress?.Report(layout.Diagnostic(region.Id));

            if (!layout.ShouldRender)
                continue;

            foreach (var line in block.Lines)
            {
                token.ThrowIfCancellationRequested();

                bool detached =
                    layout.Detected &&
                    LineMaskCoverage(line, layout) < 0.20;

                if (detached)
                {
                    // 대표 말풍선 밖에 있지만 같은 번역 블록에 포함된 OCR line.
                    // NEVER.처럼 확실한 원문은 별도로 지우되, confidence가 낮은
                    // 오검출(예: 그림을 "8"로 읽은 경우)은 원본을 보호한다.
                    if (!ShouldEraseDetachedLine(line))
                    {
                        detachedKeepCount++;
                        continue;
                    }

                    AddTextOnlyMask(
                        source,
                        mask,
                        line,
                        layout,
                        constrainToLayout: false);

                    detachedEraseCount++;
                    progress?.Report(
                        $"[detached-clean] id={region.Id} text=\"{line.Text}\" confidence={line.Confidence:0.00}");
                    continue;
                }

                AddTextOnlyMask(
                    source,
                    mask,
                    line,
                    layout,
                    constrainToLayout: true);
            }
        }

        if (detachedEraseCount > 0 || detachedKeepCount > 0)
        {
            progress?.Report(
                $"분리 원문 정리 {detachedEraseCount}개 · 저신뢰 보호 {detachedKeepCount}개");
        }

        SaveDebugOverlay(
            source,
            regions,
            layouts,
            debugPath);

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
        BalloonLayout layout,
        bool constrainToLayout)
    {
        int expand = Math.Max(1, (int)Math.Ceiling(line.H * 0.035));

        var textRect = ClampRect(
            (int)Math.Floor(line.X) - expand,
            (int)Math.Floor(line.Y) - expand,
            (int)Math.Ceiling(line.W) + expand * 2,
            (int)Math.Ceiling(line.H) + expand * 2,
            source.Cols,
            source.Rows);

        if (constrainToLayout && layout.Detected)
        {
            var clipped = Intersect(textRect, layout.Bounds);
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

        if (constrainToLayout && layout.Detected)
        {
            var clipped = Intersect(sampleRect, layout.Bounds);
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
                if (constrainToLayout && layout.Detected && !layout.Contains(x, y))
                    continue;

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
                    if (layout.Detected && !layout.Contains(x, y))
                        continue;

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

    static bool ShouldEraseDetachedLine(OcrLine line)
    {
        string text = line.Text.Trim();
        if (string.IsNullOrWhiteSpace(text))
            return false;

        int letters = text.Count(char.IsLetter);
        int digits = text.Count(char.IsDigit);

        // 일반 텍스트는 0.82 이상, 한 글자/아주 짧은 대사는 0.90 이상일 때만 지운다.
        // 숫자만 있는 detached OCR은 패널 그림/텍스처 오인 가능성이 커서 보호한다.
        if (letters >= 2)
            return line.Confidence >= 0.82f;

        if (letters == 1 && digits == 0)
            return line.Confidence >= 0.90f;

        return false;
    }

    static double LineMaskCoverage(
        OcrLine line,
        BalloonLayout layout)
    {
        if (!layout.Detected)
            return 1.0;

        int inside = 0;
        int total = 0;

        // line 전체를 촘촘히 순회하지 않고 5 x 3 샘플로 대표 마스크와의 소속을 판정한다.
        for (int yi = 0; yi < 3; yi++)
        {
            double ty = 0.20 + yi * 0.30;

            for (int xi = 0; xi < 5; xi++)
            {
                double tx = 0.10 + xi * 0.20;

                int x = (int)Math.Round(line.X + line.W * tx);
                int y = (int)Math.Round(line.Y + line.H * ty);

                total++;
                if (layout.Contains(x, y))
                    inside++;
            }
        }

        return total == 0
            ? 0
            : inside / (double)total;
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
        IReadOnlyDictionary<int, BalloonLayout> layouts,
        string outputPath,
        CancellationToken token)
    {
        var bitmap = LoadBitmap(cleanedPath);
        int width = bitmap.PixelWidth;
        int height = bitmap.PixelHeight;

        // 페이지 안에서 원문 글자 크기의 중앙값을 기준으로 잡아
        // 짧은 대사라고 글자가 갑자기 커지거나, 비슷한 캡션끼리 크기가 들쭉날쭉해지는 것을 줄인다.
        double pageFontReference = ComputePageFontReference(regions);

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

                if (!layouts.TryGetValue(region.Id, out var layout))
                {
                    var fallback = ClampRect(
                        (int)region.Source.X,
                        (int)region.Source.Y,
                        (int)region.Source.W,
                        (int)region.Source.H,
                        width,
                        height);

                    layout = new BalloonLayout(
                        fallback,
                        fallback,
                        false,
                        false,
                        null,
                        0,
                        0,
                        region.Type,
                        "rejected",
                        false,
                        "layout_missing",
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0);
                }

                if (!layout.ShouldRender)
                    continue;

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
                    region.Type,
                    ComputeSourceGlyphHint(region.Source, pageFontReference),
                    pageFontReference);
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

    static void SaveDebugOverlay(
        Mat source,
        IReadOnlyList<VisionTranslation> regions,
        IReadOnlyDictionary<int, BalloonLayout> layouts,
        string debugPath)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(debugPath)!);
            using var debug = source.Clone();

            foreach (var region in regions)
            {
                if (!layouts.TryGetValue(region.Id, out var layout))
                    continue;

                Scalar color = layout.Mode switch
                {
                    "speech" => new Scalar(0, 220, 0),
                    "caption" or "caption_flood" => new Scalar(220, 120, 0),
                    "fallback" => new Scalar(0, 220, 220),
                    _ => new Scalar(0, 0, 230)
                };

                var ocr = ClampRect(
                    (int)Math.Floor(region.Source.X),
                    (int)Math.Floor(region.Source.Y),
                    (int)Math.Ceiling(region.Source.W),
                    (int)Math.Ceiling(region.Source.H),
                    source.Cols,
                    source.Rows);

                Cv2.Rectangle(
                    debug,
                    ocr,
                    new Scalar(180, 180, 180),
                    1);

                Cv2.Rectangle(
                    debug,
                    layout.Bounds,
                    color,
                    2);

                if (layout.ShouldRender)
                {
                    Cv2.Rectangle(
                        debug,
                        layout.Inner,
                        color,
                        1);
                }

                if (layout.ShouldRender && layout.Detected)
                {
                    foreach (var line in region.Source.Lines)
                    {
                        if (LineMaskCoverage(line, layout) >= 0.20)
                            continue;

                        bool clean = ShouldEraseDetachedLine(line);
                        var lineRect = ClampRect(
                            (int)Math.Floor(line.X),
                            (int)Math.Floor(line.Y),
                            (int)Math.Ceiling(line.W),
                            (int)Math.Ceiling(line.H),
                            source.Cols,
                            source.Rows);

                        Cv2.Rectangle(
                            debug,
                            lineRect,
                            clean
                                ? new Scalar(220, 0, 220)
                                : new Scalar(0, 140, 255),
                            2);

                        if (clean)
                        {
                            Cv2.PutText(
                                debug,
                                "CLEAN",
                                new OpenCvSharp.Point(
                                    lineRect.X,
                                    Math.Max(12, lineRect.Y - 3)),
                                HersheyFonts.HersheySimplex,
                                0.36,
                                new Scalar(220, 0, 220),
                                1,
                                LineTypes.AntiAlias);
                        }
                    }
                }

                string label =
                    $"{region.Id}:{layout.Mode}" +
                    (layout.ShouldRender ? "" : ":KEEP");

                var labelPoint = new OpenCvSharp.Point(
                    Math.Max(0, layout.Bounds.X),
                    Math.Max(14, layout.Bounds.Y - 4));

                Cv2.PutText(
                    debug,
                    label,
                    labelPoint,
                    HersheyFonts.HersheySimplex,
                    0.42,
                    color,
                    1,
                    LineTypes.AntiAlias);
            }

            Cv2.ImWrite(debugPath, debug);
        }
        catch
        {
            // 디버그 출력 실패는 실제 번역/조판 작업을 중단시키지 않는다.
        }
    }

    static List<VisionTranslation> SuppressDuplicateRegions(
        IReadOnlyList<VisionTranslation> regions,
        out List<VisionTranslation> suppressed)
    {
        suppressed = [];
        if (regions.Count <= 1)
            return regions.ToList();

        // 더 완전한 블록을 먼저 검토한다. 같은 캡션/말풍선을 OCR 멀티패스가
        // 일부 줄만 다시 잡은 경우 작은 중복 블록이 뒤에서 자연스럽게 제거된다.
        var ordered = regions
            .OrderByDescending(DuplicateKeepScore)
            .ThenBy(x => x.Id)
            .ToList();

        var kept = new List<VisionTranslation>(ordered.Count);

        foreach (var candidate in ordered)
        {
            bool duplicate = kept.Any(existing =>
                IsDuplicateRenderRegion(existing, candidate));

            if (duplicate)
            {
                suppressed.Add(candidate);
                continue;
            }

            kept.Add(candidate);
        }

        return kept
            .OrderBy(x => x.Id)
            .ToList();
    }

    static double DuplicateKeepScore(VisionTranslation region)
    {
        double area = Math.Max(
            1.0,
            region.Source.W * region.Source.H);

        int textLength = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(region.CorrectedText)
                ? region.Source.Text
                : region.CorrectedText).Length;

        return
            region.Source.OriginalRegionCount * 1000.0 +
            textLength * 12.0 +
            Math.Sqrt(area);
    }

    static bool IsDuplicateRenderRegion(
        VisionTranslation a,
        VisionTranslation b)
    {
        // dialogue와 caption처럼 의미가 다른 타입끼리는 중복으로 지우지 않는다.
        if (!string.Equals(
                a.Type,
                b.Type,
                StringComparison.OrdinalIgnoreCase))
            return false;

        var ar = new OpenCvSharp.Rect2d(
            a.Source.X,
            a.Source.Y,
            Math.Max(1, a.Source.W),
            Math.Max(1, a.Source.H));

        var br = new OpenCvSharp.Rect2d(
            b.Source.X,
            b.Source.Y,
            Math.Max(1, b.Source.W),
            Math.Max(1, b.Source.H));

        double intersection = IntersectionArea(ar, br);
        double smallerArea = Math.Max(
            1.0,
            Math.Min(ar.Width * ar.Height, br.Width * br.Height));

        double containment = intersection / smallerArea;
        if (containment < 0.82)
            return false;

        string at = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(a.CorrectedText)
                ? a.Source.Text
                : a.CorrectedText);

        string bt = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(b.CorrectedText)
                ? b.Source.Text
                : b.CorrectedText);

        if (at.Length < 4 || bt.Length < 4)
            return false;

        // 큰 블록 문장 안에 작은 OCR 재검출 문장이 포함되는 전형적인 중복.
        if (at.Contains(bt, StringComparison.Ordinal) ||
            bt.Contains(at, StringComparison.Ordinal))
            return true;

        // OCR 교정 차이로 완전한 substring이 아니어도 단어 대부분이 같으면 중복.
        double tokenSimilarity = TokenContainment(at, bt);
        return containment >= 0.90 && tokenSimilarity >= 0.78;
    }

    static string NormalizeDuplicateText(string text)
    {
        var chars = text
            .ToLowerInvariant()
            .Where(char.IsLetterOrDigit)
            .ToArray();

        return new string(chars);
    }

    static double TokenContainment(string a, string b)
    {
        static HashSet<string> Tokens(string value)
        {
            var tokens = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < value.Length - 2; i++)
                tokens.Add(value.Substring(i, 3));

            return tokens;
        }

        var ta = Tokens(a);
        var tb = Tokens(b);

        if (ta.Count == 0 || tb.Count == 0)
            return 0;

        int common = ta.Count(x => tb.Contains(x));
        return common / (double)Math.Min(ta.Count, tb.Count);
    }

    static double IntersectionArea(
        OpenCvSharp.Rect2d a,
        OpenCvSharp.Rect2d b)
    {
        double left = Math.Max(a.Left, b.Left);
        double top = Math.Max(a.Top, b.Top);
        double right = Math.Min(a.Right, b.Right);
        double bottom = Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               Math.Max(0, bottom - top);
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

    static double ComputePageFontReference(
        IReadOnlyList<VisionTranslation> regions)
    {
        var samples = regions
            .Where(x => x.Render)
            .SelectMany(x => x.Source.Lines)
            .Where(x => x.Confidence >= 0.70f)
            .Select(x => Math.Min(x.W, x.H))
            .Where(x => x >= 10 && x <= 120)
            .OrderBy(x => x)
            .ToArray();

        if (samples.Length == 0)
            return 32;

        int mid = samples.Length / 2;
        return samples.Length % 2 == 1
            ? samples[mid]
            : (samples[mid - 1] + samples[mid]) / 2.0;
    }

    static double ComputeSourceGlyphHint(
        OcrTextBlock block,
        double fallback)
    {
        var samples = block.Lines
            .Where(x => x.Confidence >= 0.60f)
            .Select(x => Math.Min(x.W, x.H))
            .Where(x => x >= 8 && x <= 160)
            .OrderBy(x => x)
            .ToArray();

        if (samples.Length == 0)
            return fallback;

        int mid = samples.Length / 2;
        return samples.Length % 2 == 1
            ? samples[mid]
            : (samples[mid - 1] + samples[mid]) / 2.0;
    }

    static void DrawFittedText(
        DrawingContext dc,
        string text,
        System.Windows.Rect box,
        bool darkBackground,
        string type,
        double sourceGlyphHint,
        double pageFontReference)
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

        double containerMaxFont = Math.Clamp(
            caption
                ? Math.Min(box.Height * 0.34, 48)
                : Math.Min(box.Height * 0.46, 56),
            Math.Max(minFont, 12),
            caption ? 48 : 56);

        // 원문 OCR 글자 크기를 그대로 쓰면 페이지마다 OCR bbox 편차가 커서 불안정하다.
        // 페이지 중앙값 주변으로만 허용한 뒤 최대 폰트 크기의 상한으로 사용한다.
        double normalizedGlyph = Math.Clamp(
            sourceGlyphHint,
            pageFontReference * 0.78,
            pageFontReference * 1.22);

        double sourceDrivenMax = normalizedGlyph * (caption ? 0.92 : 0.98);

        double maxFont = Math.Min(
            containerMaxFont,
            Math.Clamp(
                sourceDrivenMax,
                Math.Max(minFont, 12),
                caption ? 46 : 52));

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

        // 말풍선/캡션 색상에 관계없이 가장 범용적으로 읽히는 기본 스타일.
        // 검은 글자 + 얇은 흰 테두리는 흰색, 컬러, 어두운 말풍선 모두에서
        // 글자 형태를 안정적으로 유지한다.
        Brush fill = Brushes.Black;
        Brush outline = Brushes.White;

        var geometry = best.BuildGeometry(origin);
        var pen = new Pen(
            outline,
            Math.Clamp(bestSize * 0.035, 0.45, 1.25));

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
