using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2EraseSelection(
    IReadOnlySet<string> TextRegionIds,
    IReadOnlySet<int> TranslationRegionIds);

/// <summary>
/// Chooses which immutable RT-DETR TextBubble targets may be erased for the
/// current translated page. Translation/OCR can decide whether a target is
/// needed, but never changes the target geometry itself.
/// </summary>
public static class V2EraseSelector
{
    public static V2EraseSelection Select(
        V2DetectionSnapshot snapshot,
        IReadOnlyList<VisionTranslation> translations)
    {
        var targetIds = new HashSet<string>(StringComparer.Ordinal);
        var regionIds = new HashSet<int>();

        foreach (var region in translations.Where(x =>
                     x.Render &&
                     !string.IsNullOrWhiteSpace(x.Translation)))
        {
            var matched = MatchTargets(
                snapshot.TextTargets,
                region);

            if (matched.Count == 0)
                continue;

            foreach (var target in matched)
                targetIds.Add(target.TextRegionId);

            regionIds.Add(region.Id);
        }

        return new V2EraseSelection(
            targetIds,
            regionIds);
    }

    static List<V2TextTarget> MatchTargets(
        IReadOnlyList<V2TextTarget> targets,
        VisionTranslation region)
    {
        // Best case: the runtime RT-DETR text-region link survived OCR/Vision.
        if (region.Source.RegionTextRegion is { } linked)
        {
            var direct = targets
                .Where(x => string.Equals(
                    x.TextRegionId,
                    linked.RegionId,
                    StringComparison.Ordinal))
                .ToList();

            if (direct.Count > 0)
                return direct;
        }

        // A recovered OCR unit may only retain the learned Bubble id. In that
        // case all TextBubble targets belonging to the same Bubble are the
        // immutable erase targets for that translated bubble.
        if (!string.IsNullOrWhiteSpace(region.Source.RegionId))
        {
            var sameBubble = targets
                .Where(x => string.Equals(
                    x.BubbleRegionId,
                    region.Source.RegionId,
                    StringComparison.Ordinal))
                .ToList();

            if (sameBubble.Count > 0)
                return sameBubble;
        }

        // Last resort: use the translation block only to select a detector
        // target. The selected target's RT-DETR bounds remain unchanged.
        var source = new Rect2d(
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
            .OrderByDescending(x => x.CenterRelated)
            .ThenByDescending(x => x.Coverage)
            .Take(2)
            .Select(x => x.Target)
            .ToList();
    }

    static double OverlapOverSmaller(
        Rect2d a,
        Rect b)
    {
        double left = Math.Max(a.Left, b.Left);
        double top = Math.Max(a.Top, b.Top);
        double right = Math.Min(a.Right, b.Right);
        double bottom = Math.Min(a.Bottom, b.Bottom);

        double intersection =
            Math.Max(0, right - left) *
            Math.Max(0, bottom - top);

        double smaller =
            Math.Max(
                1.0,
                Math.Min(
                    a.Width * a.Height,
                    b.Width * (double)b.Height));

        return intersection / smaller;
    }

    static bool ContainsCenter(
        Rect2d inner,
        Rect outer)
    {
        double cx = inner.X + inner.Width / 2.0;
        double cy = inner.Y + inner.Height / 2.0;

        return cx >= outer.Left &&
               cx <= outer.Right &&
               cy >= outer.Top &&
               cy <= outer.Bottom;
    }

    static bool ContainsCenter(
        Rect inner,
        Rect2d outer)
    {
        double cx = inner.X + inner.Width / 2.0;
        double cy = inner.Y + inner.Height / 2.0;

        return cx >= outer.Left &&
               cx <= outer.Right &&
               cy >= outer.Top &&
               cy <= outer.Bottom;
    }
}
