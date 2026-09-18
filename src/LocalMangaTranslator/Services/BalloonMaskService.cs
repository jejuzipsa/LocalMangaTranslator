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
    string Type)
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
}

public static class BalloonMaskService
{
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

        var detected = DetectMask(source, blockRect, type);

        if (detected is null)
        {
            bool caption = string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);
            int fx = Math.Max(4, (int)Math.Ceiling(blockRect.Width * (caption ? 0.08 : 0.12)));
            int fy = Math.Max(4, (int)Math.Ceiling(blockRect.Height * (caption ? 0.10 : 0.16)));

            var fallback = ClampRect(
                blockRect.X - fx,
                blockRect.Y - fy,
                blockRect.Width + fx * 2,
                blockRect.Height + fy * 2,
                source.Cols,
                source.Rows);

            bool dark = EstimateDarkBackground(source, fallback, block.Lines);

            return new BalloonLayout(
                fallback,
                fallback,
                false,
                dark,
                null,
                0,
                0,
                type);
        }

        var (bounds, detectedMask) = detected.Value;
        using var safeMask = detectedMask;
        var innerLocal = FindLargestRectangle(safeMask);

        if (innerLocal.Width < 12 || innerLocal.Height < 12)
        {
            innerLocal = new Rect(
                Math.Max(0, blockRect.X - bounds.X),
                Math.Max(0, blockRect.Y - bounds.Y),
                Math.Min(bounds.Width, Math.Max(12, blockRect.Width)),
                Math.Min(bounds.Height, Math.Max(12, blockRect.Height)));
        }

        var inner = ClampRect(
            bounds.X + innerLocal.X,
            bounds.Y + innerLocal.Y,
            innerLocal.Width,
            innerLocal.Height,
            source.Cols,
            source.Rows);

        bool darkBackground = EstimateDarkBackground(
            source,
            inner,
            block.Lines);

        return new BalloonLayout(
            bounds,
            inner,
            true,
            darkBackground,
            ToByteArray(safeMask),
            safeMask.Cols,
            safeMask.Rows,
            type);
    }

    static (Rect Bounds, Mat SafeMask)? DetectMask(
        Mat source,
        Rect blockRect,
        string type)
    {
        // StoneCandy의 balloon-fill 방식처럼 OCR 박스를 중심으로 넓은 창을 만든 뒤,
        // 큰 경계 후보를 하나씩 장벽으로 세우고 중심 flood-fill 영역이 말풍선인지 찾는다.
        var search = EnlargeWindow(
            blockRect,
            source.Cols,
            source.Rows,
            string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase) ? 2.1 : 2.5);

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
        double imageArea = Math.Max(1.0, width * (double)height);

        using var blurred = new Mat();
        using var gray = new Mat();
        using var edges = new Mat();

        Cv2.GaussianBlur(working, blurred, new Size(3, 3), 0);
        Cv2.CvtColor(blurred, gray, ColorConversionCodes.BGR2GRAY);
        Cv2.Canny(gray, edges, 70, 140, apertureSize: 3, L2gradient: true);

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

        using var barrier = Mat.Zeros(height, width, MatType.CV_8UC1).ToMat();
        Mat? bestFlood = null;
        int bestArea = int.MaxValue;

        var seed = new Point(width / 2, height / 2);
        var diff = new Scalar(10);

        try
        {
            for (int i = 0; i < contours.Length; i++)
            {
                var contourRect = Cv2.BoundingRect(contours[i]);
                if (contourRect.Width * (double)contourRect.Height < imageArea * 0.40)
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

                if (filled <= imageArea * 0.30)
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

            using (var kernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new Size(3, 3)))
            {
                Cv2.MorphologyEx(
                    rawMask,
                    rawMask,
                    MorphTypes.Close,
                    kernel,
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
            Cv2.FindNonZero(originalScaleMask, nonZero);
            if (nonZero.Empty())
                return null;

            var localBounds = Cv2.BoundingRect(nonZero);

            double blockArea = Math.Max(1.0, blockRect.Width * (double)blockRect.Height);
            double maskRectArea = Math.Max(1.0, localBounds.Width * (double)localBounds.Height);
            double ratio = maskRectArea / blockArea;

            // 패널 전체나 배경을 말풍선으로 오인했으면 사용하지 않는다.
            if (ratio > 24.0)
                return null;

            var globalBounds = new Rect(
                search.X + localBounds.X,
                search.Y + localBounds.Y,
                localBounds.Width,
                localBounds.Height);

            double coverage = IntersectionArea(globalBounds, blockRect) / blockArea;
            if (coverage < 0.55)
                return null;

            using var cropped = new Mat(originalScaleMask, localBounds);
            var safeMask = cropped.Clone();

            // 외곽선과 말풍선 꼬리를 건드리지 않도록 안쪽으로 조금 줄인 마스크를 사용한다.
            int minDim = Math.Min(safeMask.Cols, safeMask.Rows);
            int guard = Math.Clamp(minDim / 34, 2, 8);
            int kernelSize = guard * 2 + 1;

            using (var erodeKernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new Size(kernelSize, kernelSize)))
            {
                Cv2.Erode(safeMask, safeMask, erodeKernel, iterations: 1);
            }

            double safeArea = Cv2.CountNonZero(safeMask);
            if (safeArea < blockArea * 0.60)
            {
                safeMask.Dispose();
                return null;
            }

            return (globalBounds, safeMask);
        }
        finally
        {
            bestFlood?.Dispose();
        }
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

        // StoneCandy enlarge_window과 같은 면적 비율 확장식.
        double a = Math.Max(0.05, aspect);
        double b = width + height * aspect;
        double cc = (1.0 - areaRatio) * width * height;
        double discriminant = Math.Max(0, b * b - 4.0 * a * cc);
        double positiveRoot = (-b + Math.Sqrt(discriminant)) / (2.0 * a);

        int deltaY = Math.Max(18, (int)Math.Round(positiveRoot / 2.0));
        int deltaX = Math.Max(18, (int)Math.Round(deltaY * aspect));

        deltaX = Math.Min(
            deltaX,
            Math.Min(rect.X, Math.Max(0, imageWidth - rect.Right)));

        deltaY = Math.Min(
            deltaY,
            Math.Min(rect.Y, Math.Max(0, imageHeight - rect.Bottom)));

        // 이미지 가장자리 말풍선도 있으므로 한쪽 여백이 0이라고 전체 확장을 막지 않는다.
        int left = Math.Max(0, rect.X - Math.Max(18, deltaX));
        int top = Math.Max(0, rect.Y - Math.Max(18, deltaY));
        int right = Math.Min(imageWidth, rect.Right + Math.Max(18, deltaX));
        int bottom = Math.Min(imageHeight, rect.Bottom + Math.Max(18, deltaY));

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
                heights[x] = binary.At<byte>(y, x) != 0 ? heights[x] + 1 : 0;

            var stack = new Stack<int>();
            int i = 0;

            while (i <= width)
            {
                int current = i == width ? 0 : heights[i];

                if (stack.Count == 0 || current >= heights[stack.Peek()])
                {
                    stack.Push(i++);
                    continue;
                }

                int top = stack.Pop();
                int h = heights[top];
                int left = stack.Count == 0 ? 0 : stack.Peek() + 1;
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

    static bool PointInPolygon(
        IReadOnlyList<Point> polygon,
        double x,
        double y)
    {
        bool inside = false;

        for (int i = 0, j = polygon.Count - 1; i < polygon.Count; j = i++)
        {
            double xi = polygon[i].X;
            double yi = polygon[i].Y;
            double xj = polygon[j].X;
            double yj = polygon[j].Y;

            bool intersects =
                ((yi > y) != (yj > y)) &&
                (x < (xj - xi) * (y - yi) / Math.Max(1e-9, yj - yi) + xi);

            if (intersects)
                inside = !inside;
        }

        return inside;
    }

    static double IntersectionArea(Rect a, Rect b)
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
        int left = Math.Clamp(x, 0, Math.Max(0, imageWidth - 1));
        int top = Math.Clamp(y, 0, Math.Max(0, imageHeight - 1));
        int right = Math.Clamp(x + Math.Max(1, width), left + 1, imageWidth);
        int bottom = Math.Clamp(y + Math.Max(1, height), top + 1, imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }
}
