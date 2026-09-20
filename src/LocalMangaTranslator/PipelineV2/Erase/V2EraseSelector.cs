using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2RegionBinding(
    int TranslationRegionId,
    IReadOnlyList<string> TextRegionIds,
    Rect TextBounds,
    string? BubbleRegionId,
    Rect LayoutBounds,
    string LayoutMode);

public sealed record V2EraseSelection(
    IReadOnlySet<string> TextRegionIds,
    IReadOnlySet<int> TranslationRegionIds,
    IReadOnlyDictionary<int, V2RegionBinding> Bindings)
{
    public IReadOnlyDictionary<int, string> PreservationReasons { get; init; } =
        new Dictionary<int, string>();
}

/// <summary>
/// Binds translation units to immutable RT-DETR geometry.
///
/// V2 rule:
/// - TextBubble geometry owns OCR/erase/review and its parent Bubble owns layout.
/// - TextFree owns both erase and a conservative local layout rectangle.
/// - downstream code may not rediscover or replace detector geometry.
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

        var preservationReasons =
            new Dictionary<int, string>();

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

            if (ShouldPreserveStylizedGraphic(
                    region,
                    matched))
            {
                preservationReasons[region.Id] =
                    "stylized_graphic";
                continue;
            }

            foreach (var target in matched)
            {
                targetIds.Add(
                    target.TextRegionId);
            }

            var textBounds =
                UnionBounds(
                    matched.Select(x =>
                        x.TextBounds));

            var bubbleIds =
                matched
                    .Select(x => x.BubbleRegionId)
                    .Where(x =>
                        !string.IsNullOrWhiteSpace(x))
                    .Distinct(StringComparer.Ordinal)
                    .ToArray();

            var bubbleBounds =
                matched
                    .Where(x =>
                        x.BubbleBounds.HasValue)
                    .Select(x =>
                        x.BubbleBounds!.Value)
                    .Distinct()
                    .ToArray();

            bool oneParentBubble =
                bubbleIds.Length == 1 &&
                bubbleBounds.Length == 1;

            bool allTextFree =
                matched.All(x =>
                    x.Kind ==
                    PageRegionKind.TextFree);

            Rect layoutBounds =
                oneParentBubble
                    ? bubbleBounds[0]
                    : textBounds;

            string layoutMode =
                oneParentBubble
                    ? "rtdetr_parent_bubble"
                    : allTextFree
                        ? "rtdetr_textfree"
                        : "textbubble_fallback";

            regionIds.Add(
                region.Id);

            bindings[region.Id] =
                new V2RegionBinding(
                    region.Id,
                    matched
                        .Select(x =>
                            x.TextRegionId)
                        .Distinct(StringComparer.Ordinal)
                        .ToArray(),
                    textBounds,
                    oneParentBubble
                        ? bubbleIds[0]
                        : null,
                    layoutBounds,
                    layoutMode);
        }

        return new V2EraseSelection(
            targetIds,
            regionIds,
            bindings)
        {
            PreservationReasons =
                preservationReasons
        };
    }

    static bool ShouldPreserveStylizedGraphic(
        VisionTranslation region,
        IReadOnlyList<V2TextTarget> matched)
    {
        // Large outlined/colored shout lettering is part of the artwork.
        // Preserve only free-standing dialogue without a detected parent
        // Bubble. Captions/narration and ordinary Bubble dialogue stay on the
        // translation path.
        if (matched.Any(x =>
                x.BubbleBounds.HasValue))
        {
            return false;
        }

        if (!string.Equals(
                region.Type,
                "dialogue",
                StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        string text =
            string.IsNullOrWhiteSpace(
                region.CorrectedText)
                ? region.Source.Text
                : region.CorrectedText;

        var letters =
            text
                .Where(char.IsLetter)
                .ToArray();

        int meaningful =
            text.Count(
                char.IsLetterOrDigit);

        if (letters.Length < 2 ||
            meaningful < 2 ||
            meaningful > 24)
        {
            return false;
        }

        int upper =
            letters.Count(
                char.IsUpper);

        double upperRatio =
            upper /
            (double)letters.Length;

        bool emphaticPunctuation =
            text.Contains('!') ||
            text.Contains('?');

        bool compactShout =
            !text.Any(char.IsWhiteSpace) &&
            meaningful <= 12;

        return upperRatio >= 0.85 &&
               (emphaticPunctuation ||
                compactShout);
    }

    static List<V2TextTarget> MatchTargets(
        IReadOnlyList<V2TextTarget> targets,
        VisionTranslation region)
    {
        // Best case: OCR/Vision still carries the exact RT-DETR TextBubble id.
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

        // A recovered unit may only retain the learned Bubble id.
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
        // never redefine that target's TextBubble/Bubble geometry.
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
            .Select(x =>
                x.Target)
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
            list.Min(x =>
                x.Left);

        int top =
            list.Min(x =>
                x.Top);

        int right =
            list.Max(x =>
                x.Right);

        int bottom =
            list.Max(x =>
                x.Bottom);

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
