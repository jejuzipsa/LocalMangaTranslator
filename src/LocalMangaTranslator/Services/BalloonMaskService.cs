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
        bool caption = string.Equals(type, "caption", StringComparison.OrdinalIgnoreCase);

        int padX = (int)Math.Clamp(
            blockRect.Width * (caption ? 0.75 : 1.35) + blockRect.Height * 0.12,
            34,
            300);

        int padY = (int)Math.Clamp(
            blockRect.Height * (caption ? 0.85 : 1.55) + blockRect.Width * 0.10,
            32,
            260);

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
        Cv2.GaussianBlur(gray, blurred, new Size(3, 3), 0);
        Cv2.Canny(blurred, edges, 55, 135, apertureSize: 3, L2gradient: true);

        using (var closeKernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(5, 5)))
        {
            Cv2.MorphologyEx(
                edges,
                edges,
                MorphTypes.Close,
                closeKernel,
                iterations: 2);
        }

        Cv2.FindContours(
            edges,
            out Point[][] contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        double blockArea = Math.Max(1.0, blockRect.Width * (double)blockRect.Height);
        double centerX = blockRect.X + blockRect.Width / 2.0;
        double centerY = blockRect.Y + blockRect.Height / 2.0;
        double localCenterX = centerX - search.X;
        double localCenterY = centerY - search.Y;

        Point[]? bestContour = null;
        Rect bestLocalBounds = default;
        double bestScore = double.NegativeInfinity;

        foreach (var contour in contours)
        {
            if (contour.Length < 6)
                continue;

            var localBounds = Cv2.BoundingRect(contour);
            if (localBounds.Width < Math.Max(14, blockRect.Width * 0.9) ||
                localBounds.Height < Math.Max(12, blockRect.Height * 0.9))
                continue;

            var globalBounds = new Rect(
                search.X + localBounds.X,
                search.Y + localBounds.Y,
                localBounds.Width,
                localBounds.Height);

            double coverage = IntersectionArea(globalBounds, blockRect) / blockArea;
            if (coverage < 0.62)
                continue;

            if (!PointInPolygon(contour, localCenterX, localCenterY))
                continue;

            double rectArea = Math.Max(1.0, localBounds.Width * (double)localBounds.Height);
            double ratio = rectArea / blockArea;
            double maxRatio = caption ? 16.0 : 22.0;

            if (ratio < 1.05 || ratio > maxRatio)
                continue;

            int touches = 0;
            const int edgeTolerance = 3;
            if (localBounds.Left <= edgeTolerance) touches++;
            if (localBounds.Top <= edgeTolerance) touches++;
            if (localBounds.Right >= search.Width - edgeTolerance) touches++;
            if (localBounds.Bottom >= search.Height - edgeTolerance) touches++;
            if (touches >= 3)
                continue;

            double contourArea = Math.Abs(Cv2.ContourArea(contour));
            if (contourArea < blockArea * 0.70)
                continue;

            double fillRatio = Math.Clamp(contourArea / rectArea, 0, 1);
            double preferredRatio = caption ? 2.0 : 2.8;
            double sizePenalty = Math.Abs(Math.Log(ratio / preferredRatio));

            double centerPenalty =
                Math.Abs((globalBounds.X + globalBounds.Width / 2.0) - centerX) /
                Math.Max(1, globalBounds.Width) +
                Math.Abs((globalBounds.Y + globalBounds.Height / 2.0) - centerY) /
                Math.Max(1, globalBounds.Height);

            double score =
                coverage * 6.0 +
                fillRatio * (caption ? 1.1 : 0.45) -
                sizePenalty * 1.25 -
                centerPenalty * 0.8 -
                touches * 0.4;

            if (score > bestScore)
            {
                bestScore = score;
                bestContour = contour;
                bestLocalBounds = localBounds;
            }
        }

        if (bestContour is null || bestScore < 2.2)
            return null;

        using var wholeMask = Mat.Zeros(search.Height, search.Width, MatType.CV_8UC1).ToMat();
        Cv2.FillPoly(wholeMask, new[] { bestContour }, Scalar.White);

        var cropRect = ClampRect(
            bestLocalBounds.X,
            bestLocalBounds.Y,
            bestLocalBounds.Width,
            bestLocalBounds.Height,
            search.Width,
            search.Height);

        using var cropped = new Mat(wholeMask, cropRect);
        var safeMask = cropped.Clone();

        int minDim = Math.Min(safeMask.Cols, safeMask.Rows);
        int guard = Math.Clamp(minDim / 35, 2, 9);
        int kernelSize = guard * 2 + 1;

        using (var erodeKernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(kernelSize, kernelSize)))
        {
            Cv2.Erode(safeMask, safeMask, erodeKernel, iterations: 1);
        }

        double safeArea = Cv2.CountNonZero(safeMask);
        double blockAreaInMask = Math.Max(1, blockArea);

        if (safeArea < blockAreaInMask * 0.65)
        {
            safeMask.Dispose();
            return null;
        }

        var bounds = new Rect(
            search.X + cropRect.X,
            search.Y + cropRect.Y,
            cropRect.Width,
            cropRect.Height);

        return (bounds, safeMask);
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
