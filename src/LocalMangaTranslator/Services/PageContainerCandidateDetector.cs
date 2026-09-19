using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Shadow-mode page-wide container candidate detector.
///
/// This deliberately runs without OCR seeds. Candidates are geometry only:
/// finding one does not authorize translation, erase or rendering. The next
/// phases will validate candidates with OCR/Vision evidence before use.
/// </summary>
public sealed class PageContainerCandidateDetector
{
    sealed record RawCandidate(
        ContainerCandidateKind Kind,
        Rect Bounds,
        string Mode,
        bool Dark,
        double Score,
        double FillRatio,
        int BorderTouches,
        byte[] Mask,
        int MaskWidth,
        int MaskHeight);

    public IReadOnlyList<ContainerCandidate> Detect(
        string sourcePath,
        CancellationToken token = default)
    {
        using var source = Cv2.ImRead(
            sourcePath,
            ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "페이지 컨테이너 후보 검출을 위해 원본 이미지를 열 수 없습니다.");

        using var gray = new Mat();
        using var blurred = new Mat();

        Cv2.CvtColor(
            source,
            gray,
            ColorConversionCodes.BGR2GRAY);

        Cv2.GaussianBlur(
            gray,
            blurred,
            new Size(3, 3),
            0);

        var raw = new List<RawCandidate>();

        DetectThresholdRegions(
            blurred,
            threshold: 205,
            ThresholdTypes.Binary,
            mode: "page_bright_fill",
            dark: false,
            raw,
            token);

        DetectThresholdRegions(
            blurred,
            threshold: 58,
            ThresholdTypes.BinaryInv,
            mode: "page_dark_fill",
            dark: true,
            raw,
            token);

        DetectClosedEdges(
            blurred,
            raw,
            token);

        var canonical =
            Canonicalize(raw);

        return canonical
            .Select((x, index) =>
                new ContainerCandidate(
                    $"PC{index + 1:000}",
                    x.Kind,
                    x.Bounds,
                    x.Mode,
                    x.Dark,
                    x.Score,
                    x.FillRatio,
                    x.BorderTouches,
                    x.Mask,
                    x.MaskWidth,
                    x.MaskHeight))
            .ToList();
    }

    static void DetectThresholdRegions(
        Mat gray,
        double threshold,
        ThresholdTypes thresholdType,
        string mode,
        bool dark,
        List<RawCandidate> output,
        CancellationToken token)
    {
        using var binary = new Mat();

        Cv2.Threshold(
            gray,
            binary,
            threshold,
            255,
            thresholdType);

        using (var closeKernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Ellipse,
                   new Size(5, 5)))
        {
            Cv2.MorphologyEx(
                binary,
                binary,
                MorphTypes.Close,
                closeKernel,
                iterations: 2);
        }

        using (var openKernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Ellipse,
                   new Size(3, 3)))
        {
            Cv2.MorphologyEx(
                binary,
                binary,
                MorphTypes.Open,
                openKernel,
                iterations: 1);
        }

        Cv2.FindContours(
            binary,
            out Point[][] contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        double pageArea =
            Math.Max(
                1.0,
                gray.Cols * (double)gray.Rows);

        foreach (var contour in contours)
        {
            token.ThrowIfCancellationRequested();

            double area =
                Math.Abs(
                    Cv2.ContourArea(contour));

            if (area < pageArea * 0.00018 ||
                area > pageArea * 0.18)
            {
                continue;
            }

            var bounds =
                Cv2.BoundingRect(contour);

            if (!CandidateBoundsOk(
                    bounds,
                    gray.Cols,
                    gray.Rows))
            {
                continue;
            }

            double bboxArea =
                Math.Max(
                    1.0,
                    bounds.Width *
                    (double)bounds.Height);

            double fillRatio =
                Math.Clamp(
                    area / bboxArea,
                    0,
                    1);

            if (fillRatio < 0.34)
                continue;

            int borderTouches =
                CountImageBorderTouches(
                    bounds,
                    gray.Cols,
                    gray.Rows);

            if (borderTouches >= 2)
                continue;

            using var localMask =
                BuildLocalContourMask(
                    contour,
                    bounds);

            ErodeSafeInterior(
                localMask);

            if (Cv2.CountNonZero(localMask) <
                bboxArea * 0.18)
            {
                continue;
            }

            double areaRatio =
                bboxArea / pageArea;

            double score =
                fillRatio * 4.0 +
                Math.Max(
                    0,
                    1.6 - areaRatio * 10.0) -
                borderTouches * 0.75;

            output.Add(
                new RawCandidate(
                    ContainerCandidateKind.Speech,
                    bounds,
                    mode,
                    dark,
                    score,
                    fillRatio,
                    borderTouches,
                    ToByteArray(localMask),
                    localMask.Cols,
                    localMask.Rows));
        }
    }

    static void DetectClosedEdges(
        Mat gray,
        List<RawCandidate> output,
        CancellationToken token)
    {
        using var edges = new Mat();

        Cv2.Canny(
            gray,
            edges,
            45,
            130,
            apertureSize: 3,
            L2gradient: true);

        using (var kernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Rect,
                   new Size(5, 5)))
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
            out Point[][] contours,
            out _,
            RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        double pageArea =
            Math.Max(
                1.0,
                gray.Cols * (double)gray.Rows);

        foreach (var contour in contours)
        {
            token.ThrowIfCancellationRequested();

            double perimeter =
                Cv2.ArcLength(
                    contour,
                    true);

            if (perimeter < 50)
                continue;

            var approx =
                Cv2.ApproxPolyDP(
                    contour,
                    Math.Max(
                        2.0,
                        perimeter * 0.018),
                    true);

            if (approx.Length < 4 ||
                approx.Length > 14)
            {
                continue;
            }

            var bounds =
                Cv2.BoundingRect(approx);

            if (!CandidateBoundsOk(
                    bounds,
                    gray.Cols,
                    gray.Rows))
            {
                continue;
            }

            double bboxArea =
                Math.Max(
                    1.0,
                    bounds.Width *
                    (double)bounds.Height);

            double area =
                Math.Abs(
                    Cv2.ContourArea(approx));

            double areaRatio =
                bboxArea / pageArea;

            if (areaRatio < 0.00018 ||
                areaRatio > 0.16)
            {
                continue;
            }

            double fillRatio =
                Math.Clamp(
                    area / bboxArea,
                    0,
                    1);

            if (fillRatio < 0.42)
                continue;

            int borderTouches =
                CountImageBorderTouches(
                    bounds,
                    gray.Cols,
                    gray.Rows);

            if (borderTouches >= 2)
                continue;

            using var localMask =
                BuildLocalContourMask(
                    approx,
                    bounds);

            ErodeSafeInterior(
                localMask);

            if (Cv2.CountNonZero(localMask) <
                bboxArea * 0.16)
            {
                continue;
            }

            using var roi =
                new Mat(
                    gray,
                    bounds);

            double mean =
                Cv2.Mean(
                    roi,
                    localMask).Val0;

            bool dark =
                mean < 110;

            bool rectangleLike =
                approx.Length <= 6 &&
                fillRatio >= 0.68;

            var kind =
                rectangleLike
                    ? ContainerCandidateKind.Caption
                    : ContainerCandidateKind.Speech;

            double score =
                fillRatio * 4.0 +
                (rectangleLike ? 1.0 : 0.55) +
                Math.Max(
                    0,
                    1.2 - areaRatio * 8.0) -
                borderTouches * 0.75;

            output.Add(
                new RawCandidate(
                    kind,
                    bounds,
                    rectangleLike
                        ? "page_closed_rect"
                        : "page_closed_shape",
                    dark,
                    score,
                    fillRatio,
                    borderTouches,
                    ToByteArray(localMask),
                    localMask.Cols,
                    localMask.Rows));
        }
    }

    static bool CandidateBoundsOk(
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        if (bounds.Width < 24 ||
            bounds.Height < 18)
        {
            return false;
        }

        if (bounds.Width >
                imageWidth * 0.70 ||
            bounds.Height >
                imageHeight * 0.45)
        {
            return false;
        }

        double aspect =
            bounds.Width >= bounds.Height
                ? bounds.Width /
                  (double)Math.Max(
                      1,
                      bounds.Height)
                : bounds.Height /
                  (double)Math.Max(
                      1,
                      bounds.Width);

        return aspect <= 10.0;
    }

    static int CountImageBorderTouches(
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        const int tolerance = 3;

        int touches = 0;

        if (bounds.Left <= tolerance)
            touches++;

        if (bounds.Top <= tolerance)
            touches++;

        if (bounds.Right >=
            imageWidth - tolerance)
        {
            touches++;
        }

        if (bounds.Bottom >=
            imageHeight - tolerance)
        {
            touches++;
        }

        return touches;
    }

    static Mat BuildLocalContourMask(
        Point[] contour,
        Rect bounds)
    {
        var local =
            Mat.Zeros(
                Math.Max(
                    1,
                    bounds.Height),
                Math.Max(
                    1,
                    bounds.Width),
                MatType.CV_8UC1)
            .ToMat();

        var translated =
            contour
                .Select(p =>
                    new Point(
                        p.X - bounds.X,
                        p.Y - bounds.Y))
                .ToArray();

        Cv2.FillPoly(
            local,
            [translated],
            Scalar.White);

        return local;
    }

    static void ErodeSafeInterior(
        Mat mask)
    {
        int minDim =
            Math.Min(
                mask.Cols,
                mask.Rows);

        int guard =
            Math.Clamp(
                minDim / 48,
                1,
                5);

        using var kernel =
            Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new Size(
                    guard * 2 + 1,
                    guard * 2 + 1));

        Cv2.Erode(
            mask,
            mask,
            kernel,
            iterations: 1);
    }

    static List<RawCandidate> Canonicalize(
        IReadOnlyList<RawCandidate> input)
    {
        var ordered =
            input
                .OrderByDescending(x => x.Score)
                .ThenBy(x =>
                    x.Bounds.Width *
                    (double)x.Bounds.Height)
                .ToList();

        var result =
            new List<RawCandidate>();

        foreach (var candidate in ordered)
        {
            bool duplicate =
                result.Any(existing =>
                    SamePhysicalArea(
                        existing.Bounds,
                        candidate.Bounds));

            if (!duplicate)
                result.Add(candidate);

            if (result.Count >= 80)
                break;
        }

        return result
            .OrderBy(x => x.Bounds.Y)
            .ThenBy(x => x.Bounds.X)
            .ToList();
    }

    static bool SamePhysicalArea(
        Rect a,
        Rect b)
    {
        int left =
            Math.Max(
                a.Left,
                b.Left);

        int top =
            Math.Max(
                a.Top,
                b.Top);

        int right =
            Math.Min(
                a.Right,
                b.Right);

        int bottom =
            Math.Min(
                a.Bottom,
                b.Bottom);

        double intersection =
            Math.Max(
                0,
                right - left) *
            (double)Math.Max(
                0,
                bottom - top);

        if (intersection <= 0)
            return false;

        double areaA =
            Math.Max(
                1.0,
                a.Width *
                (double)a.Height);

        double areaB =
            Math.Max(
                1.0,
                b.Width *
                (double)b.Height);

        double containment =
            intersection /
            Math.Min(
                areaA,
                areaB);

        double union =
            areaA +
            areaB -
            intersection;

        double iou =
            union <= 0
                ? 0
                : intersection / union;

        return containment >= 0.84 ||
               iou >= 0.58;
    }

    static byte[] ToByteArray(
        Mat mat)
    {
        var bytes =
            new byte[
                mat.Rows *
                mat.Cols];

        int index = 0;

        for (int y = 0;
             y < mat.Rows;
             y++)
        {
            for (int x = 0;
                 x < mat.Cols;
                 x++)
            {
                bytes[index++] =
                    mat.At<byte>(
                        y,
                        x);
            }
        }

        return bytes;
    }
}
