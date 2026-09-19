using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Phase 2A OCR unit builder.
///
/// The legacy grouper is still used to propose local text blocks, but physical
/// containers are detected before Vision review. Raw OCR lines are then assigned
/// to one canonical container at most. Lines without a trustworthy container
/// remain available to the legacy grouper as conservative orphan units.
///
/// This keeps grouping changes separate from erase/inpaint changes.
/// </summary>
public sealed class OcrContainerUnitBuilder
{
    sealed record ContainerSeed(
        BalloonLayout Layout,
        OcrTextBlock Seed,
        double Score);

    readonly OcrBlockGrouper orphanGrouper = new();

    public OcrUnitBuildResult Build(
        string sourcePath,
        IReadOnlyList<OcrLine> rawLines,
        IReadOnlyList<OcrTextBlock> preliminaryBlocks,
        CancellationToken token = default)
    {
        if (rawLines.Count == 0)
            return new OcrUnitBuildResult([], 0, 0, 0);

        using var source = Cv2.ImRead(
            sourcePath,
            ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "컨테이너 배정을 위해 원본 이미지를 열 수 없습니다.");

        var seeds = new List<ContainerSeed>();

        foreach (var block in preliminaryBlocks)
        {
            token.ThrowIfCancellationRequested();

            // 저신뢰 한두 글자 조각은 다른 정상 대사 컨테이너에 배정될 수는 있지만,
            // 스스로 거대한 컨테이너를 만드는 seed로 사용하지 않는다.
            if (IsWeakSeed(block))
                continue;

            var layout = BalloonMaskService.Analyze(
                source,
                block,
                "dialogue");

            if (!layout.Detected)
            {
                var captionLayout = BalloonMaskService.Analyze(
                    source,
                    block,
                    "caption");

                if (captionLayout.Detected)
                    layout = captionLayout;
            }

            if (!layout.Detected ||
                layout.SafeMask is not { Length: > 0 })
                continue;

            seeds.Add(new ContainerSeed(
                layout,
                block,
                ContainerScore(layout, block)));
        }

        var containers = CanonicalizeContainers(seeds);

        var assigned = containers.ToDictionary(
            x => x,
            _ => new List<OcrLine>());

        var orphans = new List<OcrLine>();

        foreach (var line in rawLines)
        {
            token.ThrowIfCancellationRequested();

            ContainerSeed? best = null;
            double bestScore = double.NegativeInfinity;

            foreach (var container in containers)
            {
                double coverage = LineMaskCoverage(
                    line,
                    container.Layout);

                if (coverage < 0.45)
                    continue;

                double score =
                    coverage * 8.0 +
                    container.Score * 0.10 -
                    CenterDistancePenalty(
                        line,
                        container.Layout.Bounds);

                if (score > bestScore)
                {
                    bestScore = score;
                    best = container;
                }
            }

            if (best is null)
                orphans.Add(line);
            else
                assigned[best].Add(line);
        }

        var units = new List<OcrTextBlock>();

        foreach (var container in containers)
        {
            var lines = DeduplicateLines(
                assigned[container]);

            if (lines.Count == 0)
                continue;

            units.Add(ToBlock(
                -1,
                lines));
        }

        var orphanBlocks = orphanGrouper.Group(
            orphans);

        units.AddRange(orphanBlocks);

        var ordered = units
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .Select((x, id) => x with { Id = id })
            .ToList();

        int assignedLineCount =
            assigned.Values.Sum(x => x.Count);

        return new OcrUnitBuildResult(
            ordered,
            containers.Count,
            assignedLineCount,
            orphanBlocks.Count);
    }

    static bool IsWeakSeed(
        OcrTextBlock block)
    {
        int chars = MeaningfulLength(
            block.Text);

        double confidence =
            block.Lines.Count == 0
                ? 0
                : block.Lines.Average(x => x.Confidence);

        return block.OriginalRegionCount <= 1 &&
               chars <= 3 &&
               confidence < 0.72;
    }

    static List<ContainerSeed> CanonicalizeContainers(
        IReadOnlyList<ContainerSeed> seeds)
    {
        var ordered = seeds
            .OrderByDescending(x => x.Score)
            .ThenByDescending(x => x.Seed.OriginalRegionCount)
            .ToList();

        var result = new List<ContainerSeed>();

        foreach (var candidate in ordered)
        {
            bool duplicate = result.Any(existing =>
                IsSamePhysicalContainer(
                    existing.Layout.Bounds,
                    candidate.Layout.Bounds));

            if (!duplicate)
                result.Add(candidate);
        }

        return result;
    }

    static bool IsSamePhysicalContainer(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(a, b);

        if (intersection <= 0)
            return false;

        double areaA = Math.Max(
            1.0,
            a.Width * (double)a.Height);

        double areaB = Math.Max(
            1.0,
            b.Width * (double)b.Height);

        double containment =
            intersection /
            Math.Min(areaA, areaB);

        double union =
            areaA + areaB - intersection;

        double iou =
            union <= 0
                ? 0
                : intersection / union;

        return containment >= 0.82 ||
               iou >= 0.60;
    }

    static double ContainerScore(
        BalloonLayout layout,
        OcrTextBlock seed)
    {
        return
            seed.OriginalRegionCount * 4.0 +
            layout.LineContainment * 6.0 +
            layout.Coverage * 4.0 +
            Math.Clamp(layout.InnerRatio, 0, 1) * 2.0 -
            layout.TouchesBorder * 0.5 -
            Math.Max(0, layout.BBoxAreaRatio - 8.0) * 0.08;
    }

    static double LineMaskCoverage(
        OcrLine line,
        BalloonLayout layout)
    {
        if (!layout.Detected ||
            layout.SafeMask is null ||
            layout.MaskWidth <= 0 ||
            layout.MaskHeight <= 0)
            return 0;

        int inside = 0;
        const int samplesX = 5;
        const int samplesY = 3;
        int total = samplesX * samplesY;

        for (int iy = 0; iy < samplesY; iy++)
        {
            double ty =
                0.25 + iy * 0.25;

            int y = (int)Math.Round(
                line.Y + line.H * ty);

            for (int ix = 0; ix < samplesX; ix++)
            {
                double tx =
                    0.10 + ix * 0.20;

                int x = (int)Math.Round(
                    line.X + line.W * tx);

                int lx =
                    x - layout.Bounds.X;

                int ly =
                    y - layout.Bounds.Y;

                if (lx >= 0 &&
                    ly >= 0 &&
                    lx < layout.MaskWidth &&
                    ly < layout.MaskHeight &&
                    layout.SafeMask[
                        ly * layout.MaskWidth + lx] != 0)
                {
                    inside++;
                }
            }
        }

        return inside /
               (double)Math.Max(1, total);
    }

    static double CenterDistancePenalty(
        OcrLine line,
        Rect bounds)
    {
        double cx =
            line.X + line.W / 2.0;

        double cy =
            line.Y + line.H / 2.0;

        double bx =
            bounds.X + bounds.Width / 2.0;

        double by =
            bounds.Y + bounds.Height / 2.0;

        double dx =
            (cx - bx) /
            Math.Max(20, bounds.Width);

        double dy =
            (cy - by) /
            Math.Max(20, bounds.Height);

        return Math.Sqrt(
            dx * dx + dy * dy);
    }

    static List<OcrLine> DeduplicateLines(
        IReadOnlyList<OcrLine> input)
    {
        var ordered = input
            .OrderByDescending(x => x.Confidence)
            .ToList();

        var kept = new List<OcrLine>();

        foreach (var candidate in ordered)
        {
            int existing = kept.FindIndex(x =>
                IsSameLine(x, candidate));

            if (existing < 0)
                kept.Add(candidate);
        }

        return OrderLines(kept);
    }

    static bool IsSameLine(
        OcrLine a,
        OcrLine b)
    {
        double intersection =
            IntersectionArea(
                ToRect(a),
                ToRect(b));

        double minArea = Math.Max(
            1.0,
            Math.Min(
                a.W * a.H,
                b.W * b.H));

        if (intersection / minArea >= 0.72)
            return true;

        string at = Normalize(a.Text);
        string bt = Normalize(b.Text);

        if (at.Length == 0 ||
            bt.Length == 0 ||
            !string.Equals(
                at,
                bt,
                StringComparison.OrdinalIgnoreCase))
            return false;

        double acx = a.X + a.W / 2.0;
        double acy = a.Y + a.H / 2.0;
        double bcx = b.X + b.W / 2.0;
        double bcy = b.Y + b.H / 2.0;

        double distance =
            Math.Sqrt(
                Math.Pow(acx - bcx, 2) +
                Math.Pow(acy - bcy, 2));

        return distance <=
               Math.Max(
                   8,
                   Math.Min(
                       Math.Max(a.W, a.H),
                       Math.Max(b.W, b.H)) * 0.35);
    }

    static OcrTextBlock ToBlock(
        int id,
        IReadOnlyList<OcrLine> lines)
    {
        var ordered = OrderLines(lines);

        double x = ordered.Min(l => l.X);
        double y = ordered.Min(l => l.Y);
        double right = ordered.Max(l => l.X + l.W);
        double bottom = ordered.Max(l => l.Y + l.H);

        string text = string.Join(
            " ",
            ordered
                .Select(l => l.Text.Trim())
                .Where(t => !string.IsNullOrWhiteSpace(t)));

        string language = ordered
            .GroupBy(l => l.Language)
            .OrderByDescending(g => g.Count())
            .Select(g => g.Key)
            .FirstOrDefault() ?? "en";

        return new OcrTextBlock(
            id,
            x,
            y,
            Math.Max(1, right - x),
            Math.Max(1, bottom - y),
            text,
            ordered.Count,
            language,
            ordered);
    }

    static List<OcrLine> OrderLines(
        IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();

        bool vertical =
            list.Count >= 2 &&
            list.Count(IsVertical) >
            list.Count / 2;

        return vertical
            ? list
                .OrderByDescending(x => x.X)
                .ThenBy(x => x.Y)
                .ToList()
            : list
                .OrderBy(x => x.Y)
                .ThenBy(x => x.X)
                .ToList();
    }

    static bool IsVertical(
        OcrLine line)
        => line.H > line.W * 1.30;

    static Rect ToRect(
        OcrLine line)
        => new(
            (int)Math.Floor(line.X),
            (int)Math.Floor(line.Y),
            Math.Max(1, (int)Math.Ceiling(line.W)),
            Math.Max(1, (int)Math.Ceiling(line.H)));

    static double IntersectionArea(
        Rect a,
        Rect b)
    {
        int left =
            Math.Max(a.Left, b.Left);

        int top =
            Math.Max(a.Top, b.Top);

        int right =
            Math.Min(a.Right, b.Right);

        int bottom =
            Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               (double)Math.Max(0, bottom - top);
    }

    static int MeaningfulLength(
        string text)
        => text.Count(char.IsLetterOrDigit);

    static string Normalize(
        string text)
        => new(
            text
                .Where(char.IsLetterOrDigit)
                .Select(char.ToLowerInvariant)
                .ToArray());
}

