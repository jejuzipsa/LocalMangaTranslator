using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Detection;

public sealed record V2TextTarget(
    string TextRegionId,
    PageRegionKind Kind,
    Rect TextBounds,
    float TextScore,
    string? BubbleRegionId,
    Rect? BubbleBounds,
    float? BubbleScore);

public sealed record V2DetectionSnapshot(
    string SourceMode,
    IReadOnlyList<PageRegion> RawRegions,
    IReadOnlyList<V2TextTarget> TextTargets)
{
    public static V2DetectionSnapshot Create(PageAnalysisResult analysis)
    {
        var raw = analysis.Regions.ToArray();

        var bubbles = raw
            .Where(x => x.Kind == PageRegionKind.Bubble)
            .OrderByDescending(x => x.Score)
            .ToArray();

        var targets = raw
            .Where(x =>
                x.Kind is
                    PageRegionKind.TextBubble or
                    PageRegionKind.TextFree)
            .OrderBy(x => x.Bounds.Y)
            .ThenBy(x => x.Bounds.X)
            .Select(text =>
            {
                // TextBubble inherits its already-detected parent Bubble.
                // TextFree is intentionally left parentless: its own immutable
                // detector rectangle is both erase evidence and layout anchor.
                PageRegion? bubble =
                    text.Kind ==
                    PageRegionKind.TextBubble
                        ? bubbles
                            .Select(candidate => new
                            {
                                Region = candidate,
                                Coverage = Coverage(text.Bounds, candidate.Bounds),
                                CenterInside = ContainsCenter(text.Bounds, candidate.Bounds)
                            })
                            .Where(x => x.CenterInside || x.Coverage >= 0.45)
                            .OrderByDescending(x => x.CenterInside)
                            .ThenByDescending(x => x.Coverage)
                            .ThenByDescending(x => x.Region.Score)
                            .Select(x => x.Region)
                            .FirstOrDefault()
                        : null;

                return new V2TextTarget(
                    text.RegionId,
                    text.Kind,
                    text.Bounds,
                    text.Score,
                    bubble?.RegionId,
                    bubble?.Bounds,
                    bubble?.Score);
            })
            .ToArray();

        return new V2DetectionSnapshot(
            analysis.Mode,
            raw,
            targets);
    }

    static bool ContainsCenter(Rect inner, Rect outer)
    {
        double cx = inner.X + inner.Width / 2.0;
        double cy = inner.Y + inner.Height / 2.0;

        return cx >= outer.Left &&
               cx <= outer.Right &&
               cy >= outer.Top &&
               cy <= outer.Bottom;
    }

    static double Coverage(Rect inner, Rect outer)
    {
        int left = Math.Max(inner.Left, outer.Left);
        int top = Math.Max(inner.Top, outer.Top);
        int right = Math.Min(inner.Right, outer.Right);
        int bottom = Math.Min(inner.Bottom, outer.Bottom);

        double intersection =
            Math.Max(0, right - left) *
            (double)Math.Max(0, bottom - top);

        double area =
            Math.Max(1.0, inner.Width * (double)inner.Height);

        return intersection / area;
    }
}
