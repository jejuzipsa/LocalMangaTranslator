using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2RegionBinding(
    int TranslationRegionId,
    IReadOnlyList<string> TextRegionIds,
    Rect LayoutBounds);

public sealed record V2EraseSelection(
    IReadOnlySet<string> TextRegionIds,
    IReadOnlySet<int> TranslationRegionIds,
    IReadOnlyDictionary<int, V2RegionBinding> Bindings);

/// <summary>
/// Chooses immutable RT-DETR TextBubble targets for translated units.
///
/// Important V2 invariant:
/// the exact same matched TextBubble geometry is carried forward for BOTH
/// erase and typesetting. A later legacy balloon/container search is never
/// allowed to replace the layout box.
/// </summary>
public static class V2EraseSelector
{
    public static V2EraseSelection Select(
        V2DetectionSnapshot snapshot,
        IReadOnlyList<VisionTranslation> translations)
    {
        var targetIds =
            new HashSet<string>(
                StringComparer.Ordinal);

        var regionIds =
            new HashSet<int>();

        var bindings =
            new Dictionary<int, V2RegionBinding>();

        foreach (var region in translations.Where(x =>
                     x.Render &&
                     !string.IsNullOrWhiteSpace(x.Translation)))
        {
            var matched =
                MatchTargets(
                    snapshot.TextTargets,
                    region);

            if (matched.Count == 0)
                continue;

            foreach (var target in matched)
            {
                targetIds.Add(
                    target.TextRegionId);
            }

            regionIds.Add(
                region.Id);

            bindings[region.Id] =
                new V2RegionBinding(
                    region.Id,
                    matched
                        .Select(x => x.TextRegionId)
                        .Distinct(StringComparer.Ordinal)
                        .ToArray(),
                    UnionBounds(
                        matched.Select(x => x.TextBounds)));
        }

        return new V2EraseSelection(
            targetIds,
            regionIds,
            bindings);
    }

    static List<V2TextTarget> MatchTargets(
        IReadOnlyList<V2TextTarget> targets,
        VisionTranslation region)
    {
        // Best case: the runtime RT-DETR text-region link survived OCR/Vision.
        if (region.Source.RegionTextRegion is { } linked)
        {
            var direct =
                targets
                    .Where(x => string.Equals(
                        x.TextRegionId,
                        linked.RegionId,
                        StringComparison.Ordinal))
                    .ToList();

            if (direct.Count > 0)
                return direct;
        }

        // A recovered OCR unit may only retain the learned Bubble id.
        // Every matched TextBubble still remains immutable; layout uses the
        // union of those exact detector boxes instead of a new container box.
        if (!string.IsNullOrWhiteSpace(
                region.Source.RegionId))
        {
            var sameBubble =
                targets
                    .Where(x => string.Equals(
                        x.BubbleRegionId,
                        region.Source.RegionId,
                        StringComparison.Ordinal))
                    .ToList();

            if (sameBubble.Count > 0)
                return sameBubble;
        }

        // Last resort: OCR geometry may SELECT a detector target, but can
        // never replace its geometry.
        var source =
            new Rect2d(
                region.Source.X,
                region.Source.Y,
                Math.Max(1.0, region.Source.W),
                Math.Max(1.0, region.Source.H));

        return targets
            .Select(target => new
            {
                Target = target,
                Coverage = OverlapOverSmaller(
                    source,
                    target.TextBounds),
                CenterRelated =
                    ContainsCenter(
                        source,
                        target.TextBounds) ||
                    ContainsCenter(
                        target.TextBounds,
                        source)
            })
            .Where(x =>
                x.CenterRelated ||
                x.Coverage >= 0.18)
            .OrderByDescending(x =>
                x.CenterRelated)
            .ThenByDescending(x =>
                x.Coverage)
            .Take(2)
            .Select(x => x.Target)
            .ToList();
    }

    static Rect UnionBounds(
        IEnumerable<Rect> bounds)
    {
        var list =
            bounds.ToList();

        if (list.Count == 0)
            return new Rect();

        int left =
            list.Min(x => x.Left);

        int top =
            list.Min(x => x.Top);

        int right =
            list.Max(x => x.Right);

        int bottom =
            list.Max(x => x.Bottom);

        return new Rect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }

    static double OverlapOverSmaller(
        Rect2d a,
        Rect b)
    {
        double left =
            Math.Max(
                a.Left,
                b.Left);

        double top =
            Math.Max(
                a.Top,
                b.Top);

        double right =
            Math.Min(
                a.Right,
                b.Right);

        double bottom =
            Math.Min(
                a.Bottom,
                b.Bottom);

        double intersection =
            Math.Max(0, right - left) *
            Math.Max(0, bottom - top);

        double smaller =
            Math.Max(
                1.0,
                Math.Min(
                    a.Width * a.Height,
                    b.Width * (double)b.Height));

        return intersection /
               smaller;
    }

    static bool ContainsCenter(
        Rect2d inner,
        Rect outer)
    {
        double cx =
            inner.X +
            inner.Width / 2.0;

        double cy =
            inner.Y +
            inner.Height / 2.0;

        return cx >= outer.Left &&
               cx <= outer.Right &&
               cy >= outer.Top &&
               cy <= outer.Bottom;
    }

    static bool ContainsCenter(
        Rect inner,
        Rect2d outer)
    {
        double cx =
            inner.X +
            inner.Width / 2.0;

        double cy =
            inner.Y +
            inner.Height / 2.0;

        return cx >= outer.Left &&
               cx <= outer.Right &&
               cy >= outer.Top &&
               cy <= outer.Bottom;
    }
}
