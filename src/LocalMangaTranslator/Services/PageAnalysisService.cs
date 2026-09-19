using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Architecture 2.0 page analysis branch.
/// Region detection happens before and independently from OCR recognition.
/// Existing geometric detection remains available as a fallback and supplies
/// interior masks while RT-DETR contributes learned comic-layout evidence.
/// </summary>
public sealed class PageAnalysisService : IDisposable
{
    readonly PageContainerCandidateDetector legacyDetector = new();
    readonly RtdetrPageRegionAnalyzer rtdetr = new();

    public Task<PageAnalysisResult> AnalyzeAsync(
        string sourcePath,
        PipelineOptions options,
        CancellationToken token = default)
        => Task.Run(
            () => Analyze(
                sourcePath,
                options,
                token),
            token);

    PageAnalysisResult Analyze(
        string sourcePath,
        PipelineOptions options,
        CancellationToken token)
    {
        token.ThrowIfCancellationRequested();

        var legacy =
            legacyDetector.Detect(
                sourcePath,
                token)
            .ToList();

        if (options.RegionAnalysis ==
                RegionAnalysisMode.Legacy ||
            !rtdetr.IsReady)
        {
            return new PageAnalysisResult(
                [],
                legacy,
                "legacy",
                false,
                options.RegionAnalysis ==
                    RegionAnalysisMode.HybridRtdetr
                    ? "rtdetr_model_missing"
                    : null);
        }

        var regions =
            rtdetr.Analyze(
                sourcePath,
                token)
            .ToList();

        var fused =
            FuseContainers(
                legacy,
                regions,
                sourcePath,
                token);

        return new PageAnalysisResult(
            regions,
            fused,
            "hybrid_rtdetr",
            true,
            $"regions={regions.Count}; legacy={legacy.Count}; fused={fused.Count}");
    }

    public static IReadOnlyList<ContainerCandidate> FuseContainers(
        IReadOnlyList<ContainerCandidate> legacy,
        IReadOnlyList<PageRegion> regions,
        string sourcePath,
        CancellationToken token = default)
    {
        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "페이지 분석 fusion을 위해 원본 이미지를 열 수 없습니다.");

        var bubbles =
            regions
                .Where(
                    x =>
                        x.Kind ==
                        PageRegionKind.Bubble)
                .OrderByDescending(
                    x => x.Score)
                .ToList();

        var fused =
            new List<ContainerCandidate>();

        foreach (var candidate in legacy)
        {
            token.ThrowIfCancellationRequested();

            var supportingBubble =
                bubbles
                    .Select(
                        bubble => new
                        {
                            Bubble = bubble,
                            Overlap =
                                Containment(
                                    candidate.Bounds,
                                    bubble.Bounds)
                        })
                    .Where(
                        x =>
                            x.Overlap >= 0.25)
                    .OrderByDescending(
                        x => x.Overlap)
                    .ThenByDescending(
                        x => x.Bubble.Score)
                    .FirstOrDefault();

            if (supportingBubble is null)
            {
                fused.Add(
                    candidate);
                continue;
            }

            fused.Add(
                candidate with
                {
                    DetectorMode =
                        $"rtdetr+{candidate.DetectorMode}",
                    Score =
                        candidate.Score +
                        1.2 +
                        supportingBubble.Bubble.Score,
                    FillRatio =
                        Math.Clamp(
                            candidate.FillRatio +
                            0.04,
                            0,
                            1)
                });
        }

        foreach (var bubble in bubbles)
        {
            token.ThrowIfCancellationRequested();

            bool represented =
                fused.Any(
                    candidate =>
                        Containment(
                            candidate.Bounds,
                            bubble.Bounds) >=
                        0.55);

            if (represented)
                continue;

            var bounds =
                ClampRect(
                    bubble.Bounds,
                    source.Cols,
                    source.Rows);

            if (bounds.Width < 24 ||
                bounds.Height < 18)
            {
                continue;
            }

            var (mask, fillRatio) =
                BuildSafeRectangleMask(
                    bounds);

            fused.Add(
                new ContainerCandidate(
                    "",
                    ContainerCandidateKind.Speech,
                    bounds,
                    "rtdetr_bubble",
                    false,
                    4.8 +
                    bubble.Score * 2.2,
                    fillRatio,
                    CountBorderTouches(
                        bounds,
                        source.Cols,
                        source.Rows),
                    mask,
                    bounds.Width,
                    bounds.Height));
        }

        var canonical =
            Canonicalize(
                fused);

        return canonical
            .Select(
                (x, index) =>
                    x with
                    {
                        CandidateId =
                            $"PC{index + 1:000}"
                    })
            .ToList();
    }

    static (byte[] Mask, double FillRatio) BuildSafeRectangleMask(
        Rect bounds)
    {
        int width =
            Math.Max(
                1,
                bounds.Width);

        int height =
            Math.Max(
                1,
                bounds.Height);

        var mask =
            new byte[
                width *
                height];

        int guard =
            Math.Clamp(
                Math.Min(
                    width,
                    height) / 24,
                2,
                10);

        int left =
            guard;

        int top =
            guard;

        int right =
            Math.Max(
                left,
                width - guard);

        int bottom =
            Math.Max(
                top,
                height - guard);

        long count = 0;

        for (int y = top; y < bottom; y++)
        {
            for (int x = left; x < right; x++)
            {
                mask[y * width + x] =
                    255;

                count++;
            }
        }

        return (
            mask,
            count /
            (double)Math.Max(
                1,
                width * (long)height));
    }

    static List<ContainerCandidate> Canonicalize(
        IReadOnlyList<ContainerCandidate> input)
    {
        var ordered =
            input
                .OrderByDescending(
                    x => x.Score)
                .ThenByDescending(
                    x =>
                        x.Bounds.Width *
                        (long)x.Bounds.Height)
                .ToList();

        var kept =
            new List<ContainerCandidate>();

        foreach (var candidate in ordered)
        {
            int duplicate =
                kept.FindIndex(
                    existing =>
                        SamePhysicalArea(
                            existing.Bounds,
                            candidate.Bounds));

            if (duplicate < 0)
            {
                kept.Add(
                    candidate);

                continue;
            }

            // Prefer real contour masks over RT-DETR rectangle fallbacks when
            // the learned detector and geometric detector refer to the same
            // physical bubble.
            bool candidateHasContour =
                !string.Equals(
                    candidate.DetectorMode,
                    "rtdetr_bubble",
                    StringComparison.Ordinal);

            bool existingFallback =
                string.Equals(
                    kept[duplicate].DetectorMode,
                    "rtdetr_bubble",
                    StringComparison.Ordinal);

            if (candidateHasContour &&
                existingFallback)
            {
                kept[duplicate] =
                    candidate;
            }
        }

        return kept
            .OrderBy(
                x => x.Bounds.Y)
            .ThenBy(
                x => x.Bounds.X)
            .Take(100)
            .ToList();
    }

    static bool SamePhysicalArea(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        if (intersection <= 0)
            return false;

        double areaA =
            Math.Max(
                1,
                a.Width *
                (double)a.Height);

        double areaB =
            Math.Max(
                1,
                b.Width *
                (double)b.Height);

        double containment =
            intersection /
            Math.Min(
                areaA,
                areaB);

        double iou =
            intersection /
            Math.Max(
                1,
                areaA +
                areaB -
                intersection);

        return containment >= 0.82 ||
               iou >= 0.58;
    }

    static double Containment(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        if (intersection <= 0)
            return 0;

        double smaller =
            Math.Max(
                1,
                Math.Min(
                    a.Width *
                    (double)a.Height,
                    b.Width *
                    (double)b.Height));

        return intersection /
               smaller;
    }

    static double IntersectionArea(
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

        return Math.Max(
                   0,
                   right - left) *
               (double)Math.Max(
                   0,
                   bottom - top);
    }

    static Rect ClampRect(
        Rect rect,
        int imageWidth,
        int imageHeight)
    {
        int left =
            Math.Clamp(
                rect.Left,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                rect.Top,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                rect.Right,
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                rect.Bottom,
                0,
                imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(
                0,
                right - left),
            Math.Max(
                0,
                bottom - top));
    }

    static int CountBorderTouches(
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        const int tolerance =
            3;

        int touches =
            0;

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

    public void Dispose()
        => rtdetr.Dispose();
}
