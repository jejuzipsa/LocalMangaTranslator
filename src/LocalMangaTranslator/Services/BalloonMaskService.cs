using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public sealed record BalloonLayout(
    Rect Bounds,
    Rect Inner,
    bool Detected,
    bool DarkBackground,
    byte[]? SafeMask,
    int MaskWidth,
    int MaskHeight,
    string Type,
    string Mode,
    bool ShouldRender,
    string Reason,
    double MaskAreaRatio,
    double BBoxAreaRatio,
    double Coverage,
    int TouchesBorder,
    double InnerRatio,
    double LineContainment,
    double TextureStdDev)
{
    public bool Contains(int x, int y)
    {
        if (!Detected || SafeMask is null || MaskWidth <= 0 || MaskHeight <= 0)
            return true;

        int lx = x - Bounds.X;
        int ly = y - Bounds.Y;
        if (lx < 0 || ly < 0 || lx >= MaskWidth || ly >= MaskHeight)
            return false;

        return SafeMask[ly * MaskWidth + lx] != 0;
    }

    public string Diagnostic(int id)
        => $"[balloon] id={id} mode={Mode} detected={Detected} render={ShouldRender} " +
           $"mask_ratio={MaskAreaRatio:0.00} bbox_ratio={BBoxAreaRatio:0.00} " +
           $"coverage={Coverage:0.00} border={TouchesBorder} inner={InnerRatio:0.00} " +
           $"lines={LineContainment:0.00} texture={TextureStdDev:0.0} reason={Reason}";
}

public static class BalloonMaskService
{
    sealed record MaskCandidate(
        Rect Bounds,
        Rect Search,
        Mat Mask,
        string Mode);

    public static BalloonLayout Analyze(
        Mat source,
        OcrTextBlock block,
        string type)
    {
        var blockRect = ClampRect(
            (int)Math.Floor(block.X),
            (int)Math.Floor(block.Y),
            (int)Math.Ceiling(block.W),
            (int)Math.Ceiling(block.H),
            source.Cols,
            source.Rows);

        bool caption =
            string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);

        MaskCandidate? candidate = null;

        // 사각 내레이션/캡션은 일반 말풍선 flood-fill보다 먼저 전용 검출한다.
        if (caption)
            candidate = DetectCaptionMask(source, blockRect);

        candidate ??= DetectSpeechMask(source, blockRect, caption);

        if (candidate is null)
            return CreateConservativeFallback(source, block, blockRect, type, "no_mask");

        using (candidate.Mask)
        {
            var innerLocal = FindLargestRectangle(candidate.Mask);

            double blockArea = Math.Max(
                1.0,
                blockRect.Width * (double)blockRect.Height);

            double maskArea = Cv2.CountNonZero(candidate.Mask);
            double bboxArea = Math.Max(
                1.0,
                candidate.Bounds.Width * (double)candidate.Bounds.Height);

            double maskAreaRatio = maskArea / blockArea;
            double bboxAreaRatio = bboxArea / blockArea;
            double coverage = IntersectionArea(candidate.Bounds, blockRect) / blockArea;
            int touchesBorder = CountTouchesSearchBorder(candidate.Bounds, candidate.Search);

            double innerArea = Math.Max(
                0.0,
                innerLocal.Width * (double)innerLocal.Height);

            double innerRatio = maskArea > 0
                ? innerArea / maskArea
                : 0;

            double lineContainment = ComputeLineContainment(
                candidate.Mask,
                candidate.Bounds,
                block.Lines);

            double textureStdDev = ComputeTextureStdDev(
                source,
                candidate.Mask,
                candidate.Bounds,
                block.Lines);

            var reasons = new List<string>();

            double maxMaskRatio = caption ? 14.0 : 18.0;
            double maxBBoxRatio = caption ? 18.0 : 24.0;

            if (maskAreaRatio > maxMaskRatio)
                reasons.Add("too_large_mask");

            if (bboxAreaRatio > maxBBoxRatio)
                reasons.Add("too_large_bbox");

            if (coverage < 0.60)
                reasons.Add("low_coverage");

            if (touchesBorder >= 3)
                reasons.Add("touches_border");

            if (innerRatio < 0.15)
                reasons.Add("low_inner_ratio");

            if (lineContainment < 0.70)
                reasons.Add("low_line_containment");

            // 색 있는 말풍선은 허용한다. 다만 큰 영역인데 내부 질감까지 매우 복잡하면
            // 패널/배경 오탐 가능성이 높으므로 그때만 거른다.
            if (textureStdDev > 78 &&
                (maskAreaRatio > 6.0 || bboxAreaRatio > 8.0))
                reasons.Add("high_texture_variance");

            if (innerLocal.Width < 12 || innerLocal.Height < 12)
                reasons.Add("tiny_inner_rect");

            if (reasons.Count > 0)
            {
                return CreateConservativeFallback(
                    source,
                    block,
                    blockRect,
                    type,
                    string.Join(",", reasons),
                    maskAreaRatio,
                    bboxAreaRatio,
                    coverage,
                    touchesBorder,
                    innerRatio,
                    lineContainment,
                    textureStdDev,
                    rejectedCandidate: true);
            }

            var inner = ClampRect(
                candidate.Bounds.X + innerLocal.X,
                candidate.Bounds.Y + innerLocal.Y,
                innerLocal.Width,
                innerLocal.Height,
                source.Cols,
                source.Rows);

            bool dark = EstimateDarkBackground(
                source,
                inner,
                block.Lines);

            return new BalloonLayout(
                candidate.Bounds,
                inner,
                true,
                dark,
                ToByteArray(candidate.Mask),
                candidate.Mask.Cols,
                candidate.Mask.Rows,
                type,
                candidate.Mode,
                true,
                "ok",
                maskAreaRatio,
                bboxAreaRatio,
                coverage,
                touchesBorder,
                innerRatio,
                lineContainment,
                textureStdDev);
        }
    }

    static BalloonLayout CreateConservativeFallback(
        Mat source,
        OcrTextBlock block,
        Rect blockRect,
        string type,
        string reason,
        double maskAreaRatio = 0,
        double bboxAreaRatio = 0,
        double coverage = 0,
        int touchesBorder = 0,
        double innerRatio = 0,
        double lineContainment = 0,
        double textureStdDev = 0,
        bool rejectedCandidate = false)
    {
        bool caption =
            string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);

        int fx = Math.Max(
            3,
            (int)Math.Ceiling(blockRect.Width * (caption ? 0.06 : 0.08)));

        int fy = Math.Max(
            3,
            (int)Math.Ceiling(blockRect.Height * (caption ? 0.08 : 0.10)));

        var fallback = ClampRect(
            blockRect.X - fx,
            blockRect.Y - fy,
            blockRect.Width + fx * 2,
            blockRect.Height + fy * 2,
            source.Cols,
            source.Rows);

        double avgConfidence = block.Lines.Count == 0
            ? 0
            : block.Lines.Average(x => x.Confidence);

        int meaningfulLength = block.Text.Count(char.IsLetterOrDigit);

        // 말풍선 검출 실패 시 긴 문장까지 억지로 작은 OCR 박스에 조판하지 않는다.
        // 확실한 짧은 대사만 작은 fallback을 허용하고, 나머지는 원문을 유지한다.
        bool allowFallback =
            avgConfidence >= 0.52 &&
            meaningfulLength <= 46 &&
            block.OriginalRegionCount <= 4 &&
            blockRect.Width * (double)blockRect.Height <= 120000;

        bool dark = EstimateDarkBackground(
            source,
            fallback,
            block.Lines);

        string mode = allowFallback
            ? "fallback"
            : "rejected";

        string finalReason =
            rejectedCandidate
                ? $"{reason};fallback={(allowFallback ? "small" : "keep_original")}"
                : $"{reason};fallback={(allowFallback ? "small" : "keep_original")}";

        return new BalloonLayout(
            fallback,
            fallback,
            false,
            dark,
            null,
            0,
            0,
            type,
            mode,
            allowFallback,
            finalReason,
            maskAreaRatio,
            bboxAreaRatio,
            coverage,
            touchesBorder,
            innerRatio,
            lineContainment,
            textureStdDev);
    }

    static MaskCandidate? DetectCaptionMask(
        Mat source,
        Rect blockRect)
    {
        var search = ExpandRect(
            blockRect,
            source.Cols,
            source.Rows,
            0.70,
            0.85,
            28,
            220);

        using var roi = new Mat(source, search);
        using var gray = new Mat();
        using var blurred = new Mat();
        using var edges = new Mat();

        Cv2.CvtColor(roi, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Cv2.Canny(blurred, edges, 45, 125);

        using (var kernel = Cv2.GetStructuringElement(
            MorphShapes.Rect,
            new Size(5, 5)))
        {
            Cv2.MorphologyEx(
                edges,
                edges,
                MorphTypes.Close,
                kernel,
                iterations: 2);
        }

        Cv2.FindContours(
            edges,
            out Point[][] contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        double blockArea = Math.Max(
            1.0,
            blockRect.Width * (double)blockRect.Height);

        Point[]? best = null;
        Rect bestLocal = default;
        double bestScore = double.NegativeInfinity;

        foreach (var contour in contours)
        {
            double perimeter = Cv2.ArcLength(contour, true);
            if (perimeter <= 0)
                continue;

            var approx = Cv2.ApproxPolyDP(
                contour,
                Math.Max(2.0, perimeter * 0.025),
                true);

            if (approx.Length < 4 || approx.Length > 8)
                continue;

            var local = Cv2.BoundingRect(approx);
            var global = new Rect(
                search.X + local.X,
                search.Y + local.Y,
                local.Width,
                local.Height);

            double coverage = IntersectionArea(global, blockRect) / blockArea;
            if (coverage < 0.70)
                continue;

            double area = Math.Max(
                1.0,
                local.Width * (double)local.Height);

            double ratio = area / blockArea;
            if (ratio < 1.05 || ratio > 14.0)
                continue;

            double contourArea = Math.Abs(Cv2.ContourArea(contour));
            double rectangularity = Math.Clamp(contourArea / area, 0, 1);

            if (rectangularity < 0.45)
                continue;

            double score =
                coverage * 5.0 +
                rectangularity * 2.0 -
                Math.Abs(Math.Log(ratio / 1.8));

            if (score > bestScore)
            {
                bestScore = score;
                best = approx;
                bestLocal = local;
            }
        }

        if (best is null || bestScore < 3.2)
            return null;

        using var fullMask = Mat.Zeros(
            search.Height,
            search.Width,
            MatType.CV_8UC1).ToMat();

        Cv2.FillPoly(
            fullMask,
            new[] { best },
            Scalar.White);

        using var cropped = new Mat(fullMask, bestLocal);
        var safe = cropped.Clone();

        int guard = Math.Clamp(
            Math.Min(safe.Cols, safe.Rows) / 40,
            2,
            7);

        using (var erodeKernel = Cv2.GetStructuringElement(
            MorphShapes.Rect,
            new Size(guard * 2 + 1, guard * 2 + 1)))
        {
            Cv2.Erode(
                safe,
                safe,
                erodeKernel,
                iterations: 1);
        }

        var bounds = new Rect(
            search.X + bestLocal.X,
            search.Y + bestLocal.Y,
            bestLocal.Width,
            bestLocal.Height);

        return new MaskCandidate(
            bounds,
            search,
            safe,
            "caption");
    }

    static MaskCandidate? DetectSpeechMask(
        Mat source,
        Rect blockRect,
        bool caption)
    {
        var search = EnlargeWindow(
            blockRect,
            source.Cols,
            source.Rows,
            caption ? 2.1 : 2.5);

        using var roi = new Mat(source, search);

        double scale = 1.0;
        if (roi.Rows > 300 && roi.Cols > 300)
            scale = 0.60;
        else if (roi.Rows < 120 || roi.Cols < 120)
            scale = 1.40;

        using var working = new Mat();

        if (Math.Abs(scale - 1.0) > 0.01)
        {
            Cv2.Resize(
                roi,
                working,
                new Size(
                    Math.Max(8, (int)Math.Round(roi.Cols * scale)),
                    Math.Max(8, (int)Math.Round(roi.Rows * scale))),
                0,
                0,
                InterpolationFlags.Area);
        }
        else
        {
            roi.CopyTo(working);
        }

        int width = working.Cols;
        int height = working.Rows;
        double imageArea = Math.Max(
            1.0,
            width * (double)height);

        using var blurred = new Mat();
        using var gray = new Mat();
        using var edges = new Mat();

        Cv2.GaussianBlur(
            working,
            blurred,
            new Size(3, 3),
            0);

        Cv2.CvtColor(
            blurred,
            gray,
            ColorConversionCodes.BGR2GRAY);

        Cv2.Canny(
            gray,
            edges,
            70,
            140,
            apertureSize: 3,
            L2gradient: true);

        Cv2.Rectangle(
            edges,
            new Point(0, 0),
            new Point(width - 1, height - 1),
            Scalar.White,
            1,
            LineTypes.Link8);

        Cv2.FindContours(
            edges,
            out Point[][] contours,
            out _,
            RetrievalModes.CComp,
            ContourApproximationModes.ApproxNone);

        using var barrier = Mat.Zeros(
            height,
            width,
            MatType.CV_8UC1).ToMat();

        Mat? bestFlood = null;
        int bestArea = int.MaxValue;

        // 검색 ROI의 중앙보다 실제 OCR 블록 중심을 seed로 사용한다.
        var seed = new Point(
            Math.Clamp(
                (int)Math.Round(
                    (blockRect.X + blockRect.Width / 2.0 - search.X) * scale),
                1,
                Math.Max(1, width - 2)),
            Math.Clamp(
                (int)Math.Round(
                    (blockRect.Y + blockRect.Height / 2.0 - search.Y) * scale),
                1,
                Math.Max(1, height - 2)));

        var diff = new Scalar(10);

        try
        {
            for (int i = 0; i < contours.Length; i++)
            {
                var contourRect = Cv2.BoundingRect(contours[i]);

                if (contourRect.Width * (double)contourRect.Height <
                    imageArea * 0.35)
                    continue;

                Cv2.DrawContours(
                    barrier,
                    contours,
                    i,
                    Scalar.White,
                    2,
                    LineTypes.Link8);

                using var candidate = barrier.Clone();

                int filled = Cv2.FloodFill(
                    candidate,
                    seed,
                    new Scalar(127),
                    out _,
                    diff,
                    diff,
                    FloodFillFlags.Link4);

                if (filled <= imageArea * 0.20)
                {
                    Cv2.DrawContours(
                        barrier,
                        contours,
                        i,
                        Scalar.Black,
                        2,
                        LineTypes.Link8);
                    continue;
                }

                if (filled < bestArea)
                {
                    bestFlood?.Dispose();
                    bestFlood = candidate.Clone();
                    bestArea = filled;
                }
            }

            if (bestFlood is null)
                return null;

            using var rawMask = new Mat();

            Cv2.InRange(
                bestFlood,
                new Scalar(126),
                new Scalar(128),
                rawMask);

            using (var closeKernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new Size(3, 3)))
            {
                Cv2.MorphologyEx(
                    rawMask,
                    rawMask,
                    MorphTypes.Close,
                    closeKernel,
                    iterations: 1);
            }

            using var originalScaleMask = new Mat();

            if (Math.Abs(scale - 1.0) > 0.01)
            {
                Cv2.Resize(
                    rawMask,
                    originalScaleMask,
                    new Size(search.Width, search.Height),
                    0,
                    0,
                    InterpolationFlags.Nearest);
            }
            else
            {
                rawMask.CopyTo(originalScaleMask);
            }

            using var nonZero = new Mat();
            Cv2.FindNonZero(
                originalScaleMask,
                nonZero);

            if (nonZero.Empty())
                return null;

            var localBounds = Cv2.BoundingRect(nonZero);

            using var cropped = new Mat(
                originalScaleMask,
                localBounds);

            var safeMask = cropped.Clone();

            int minDim = Math.Min(
                safeMask.Cols,
                safeMask.Rows);

            int guard = Math.Clamp(
                minDim / 34,
                2,
                8);

            using (var erodeKernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new Size(guard * 2 + 1, guard * 2 + 1)))
            {
                Cv2.Erode(
                    safeMask,
                    safeMask,
                    erodeKernel,
                    iterations: 1);
            }

            var bounds = new Rect(
                search.X + localBounds.X,
                search.Y + localBounds.Y,
                localBounds.Width,
                localBounds.Height);

            return new MaskCandidate(
                bounds,
                search,
                safeMask,
                caption ? "caption_flood" : "speech");
        }
        finally
        {
            bestFlood?.Dispose();
        }
    }

    static double ComputeLineContainment(
        Mat mask,
        Rect bounds,
        IReadOnlyList<OcrLine> lines)
    {
        if (lines.Count == 0)
            return 0;

        int inside = 0;
        int total = 0;

        foreach (var line in lines)
        {
            // 한 줄에 5개 샘플을 찍어 OCR 블록 대부분이 마스크 안에 있는지 본다.
            for (int i = 0; i < 5; i++)
            {
                double t = i / 4.0;
                int x = (int)Math.Round(
                    line.X + line.W * (0.10 + 0.80 * t));

                int y = (int)Math.Round(
                    line.Y + line.H * 0.50);

                total++;

                int lx = x - bounds.X;
                int ly = y - bounds.Y;

                if (lx >= 0 &&
                    ly >= 0 &&
                    lx < mask.Cols &&
                    ly < mask.Rows &&
                    mask.At<byte>(ly, lx) != 0)
                    inside++;
            }
        }

        return total == 0
            ? 0
            : inside / (double)total;
    }

    static double ComputeTextureStdDev(
        Mat source,
        Mat mask,
        Rect bounds,
        IReadOnlyList<OcrLine> lines)
    {
        using var roi = new Mat(source, bounds);
        using var gray = new Mat();

        Cv2.CvtColor(
            roi,
            gray,
            ColorConversionCodes.BGR2GRAY);

        using var sampleMask = mask.Clone();

        // OCR 글자 자체의 검은/흰 획 때문에 분산이 높아지는 걸 막기 위해
        // 텍스트 주변은 texture 검사에서 제외한다.
        foreach (var line in lines)
        {
            var local = new Rect(
                (int)Math.Floor(line.X - bounds.X - 2),
                (int)Math.Floor(line.Y - bounds.Y - 2),
                (int)Math.Ceiling(line.W + 4),
                (int)Math.Ceiling(line.H + 4));

            local = ClampRect(
                local.X,
                local.Y,
                local.Width,
                local.Height,
                sampleMask.Cols,
                sampleMask.Rows);

            Cv2.Rectangle(
                sampleMask,
                local,
                Scalar.Black,
                -1);
        }

        if (Cv2.CountNonZero(sampleMask) < 20)
            return 0;

        Cv2.MeanStdDev(
            gray,
            out _,
            out var stddev,
            sampleMask);

        return stddev.Val0;
    }

    static int CountTouchesSearchBorder(
        Rect bounds,
        Rect search)
    {
        int tolerance = 3;
        int touches = 0;

        if (bounds.Left <= search.Left + tolerance)
            touches++;

        if (bounds.Top <= search.Top + tolerance)
            touches++;

        if (bounds.Right >= search.Right - tolerance)
            touches++;

        if (bounds.Bottom >= search.Bottom - tolerance)
            touches++;

        return touches;
    }

    static Rect ExpandRect(
        Rect rect,
        int imageWidth,
        int imageHeight,
        double factorX,
        double factorY,
        int minPad,
        int maxPad)
    {
        int px = (int)Math.Clamp(
            rect.Width * factorX,
            minPad,
            maxPad);

        int py = (int)Math.Clamp(
            rect.Height * factorY,
            minPad,
            maxPad);

        return ClampRect(
            rect.X - px,
            rect.Y - py,
            rect.Width + px * 2,
            rect.Height + py * 2,
            imageWidth,
            imageHeight);
    }

    static Rect EnlargeWindow(
        Rect rect,
        int imageWidth,
        int imageHeight,
        double areaRatio)
    {
        double width = Math.Max(1, rect.Width);
        double height = Math.Max(1, rect.Height);
        double aspect = height / width;

        double a = Math.Max(0.05, aspect);
        double b = width + height * aspect;
        double c = (1.0 - areaRatio) * width * height;
        double discriminant = Math.Max(
            0,
            b * b - 4.0 * a * c);

        double positiveRoot =
            (-b + Math.Sqrt(discriminant)) /
            (2.0 * a);

        int deltaY = Math.Max(
            18,
            (int)Math.Round(positiveRoot / 2.0));

        int deltaX = Math.Max(
            18,
            (int)Math.Round(deltaY * aspect));

        int left = Math.Max(
            0,
            rect.X - deltaX);

        int top = Math.Max(
            0,
            rect.Y - deltaY);

        int right = Math.Min(
            imageWidth,
            rect.Right + deltaX);

        int bottom = Math.Min(
            imageHeight,
            rect.Bottom + deltaY);

        return new Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }

    static Rect FindLargestRectangle(Mat binary)
    {
        if (binary.Empty())
            return new Rect(0, 0, 0, 0);

        int width = binary.Cols;
        int height = binary.Rows;
        var heights = new int[width];

        int bestArea = 0;
        Rect best = new(0, 0, 0, 0);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                heights[x] =
                    binary.At<byte>(y, x) != 0
                        ? heights[x] + 1
                        : 0;
            }

            var stack = new Stack<int>();
            int i = 0;

            while (i <= width)
            {
                int current =
                    i == width
                        ? 0
                        : heights[i];

                if (stack.Count == 0 ||
                    current >= heights[stack.Peek()])
                {
                    stack.Push(i++);
                    continue;
                }

                int top = stack.Pop();
                int h = heights[top];
                int left =
                    stack.Count == 0
                        ? 0
                        : stack.Peek() + 1;

                int w = i - left;
                int area = h * w;

                if (area > bestArea)
                {
                    bestArea = area;
                    best = new Rect(
                        left,
                        y - h + 1,
                        w,
                        h);
                }
            }
        }

        return best;
    }

    static bool EstimateDarkBackground(
        Mat source,
        Rect inner,
        IReadOnlyList<OcrLine> lines)
    {
        var luminances = new List<double>();
        int step =
            inner.Width * inner.Height > 30000
                ? 4
                : 3;

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

                var p = source.At<Vec3b>(y, x);

                luminances.Add(
                    p.Item2 * 0.299 +
                    p.Item1 * 0.587 +
                    p.Item0 * 0.114);
            }
        }

        if (luminances.Count == 0)
        {
            using var roi = new Mat(source, inner);
            var mean = Cv2.Mean(roi);

            double fallback =
                mean.Val2 * 0.299 +
                mean.Val1 * 0.587 +
                mean.Val0 * 0.114;

            return fallback < 125;
        }

        luminances.Sort();
        return luminances[luminances.Count / 2] < 125;
    }

    static byte[] ToByteArray(Mat mat)
    {
        int rows = mat.Rows;
        int cols = mat.Cols;
        var data = new byte[rows * cols];

        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
                data[y * cols + x] = mat.At<byte>(y, x);

        return data;
    }

    static double IntersectionArea(
        Rect a,
        Rect b)
    {
        int left = Math.Max(a.Left, b.Left);
        int top = Math.Max(a.Top, b.Top);
        int right = Math.Min(a.Right, b.Right);
        int bottom = Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               (double)Math.Max(0, bottom - top);
    }

    static Rect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left = Math.Clamp(
            x,
            0,
            Math.Max(0, imageWidth - 1));

        int top = Math.Clamp(
            y,
            0,
            Math.Max(0, imageHeight - 1));

        int right = Math.Clamp(
            x + Math.Max(1, width),
            left + 1,
            imageWidth);

        int bottom = Math.Clamp(
            y + Math.Max(1, height),
            top + 1,
            imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }
}
