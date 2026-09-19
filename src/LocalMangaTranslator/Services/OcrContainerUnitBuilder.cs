using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// 0007 container-first OCR unit builder.
///
/// Page-wide container candidates are canonicalized before text grouping.
/// Every merged OCR line receives at most one physical container owner.
/// Lines with ambiguous or insufficient ownership evidence remain orphans and
/// only those orphan lines fall back to the legacy proximity grouper.
/// </summary>
public sealed class OcrContainerUnitBuilder
{
    sealed class CanonicalContainer
    {
        public CanonicalContainer(ContainerCandidate candidate)
        {
            Candidate = candidate;
            SourceIds.Add(candidate.CandidateId);
        }

        public ContainerCandidate Candidate { get; }
        public HashSet<string> SourceIds { get; } = new(StringComparer.Ordinal);
    }

    sealed record CanonicalLine(string LineId, OcrLine Line);

    sealed record OwnershipCandidate(
        CanonicalContainer Container,
        double Coverage,
        double Score,
        bool HasContainerEvidence);

    sealed record UnitDraft(
        OcrTextBlock Block,
        string? CandidateId,
        IReadOnlyList<string> LineIds,
        bool IsOrphan,
        string Reason);

    readonly OcrBlockGrouper orphanGrouper = new();

    public OcrUnitBuildResult Build(
        IReadOnlyList<OcrLine> rawLines,
        IReadOnlyList<ContainerCandidate> pageCandidates,
        IReadOnlyList<ContainerOcrDecision> candidateDecisions,
        IReadOnlyList<OcrObservation> containerObservations,
        IReadOnlyList<ContainerLineDecision> lineDecisions,
        CancellationToken token = default)
    {
        if (rawLines.Count == 0)
            return new OcrUnitBuildResult([], 0, 0, 0);

        var eligibleIds = candidateDecisions
            .Where(x => x.Eligible)
            .Select(x => x.CandidateId)
            .ToHashSet(StringComparer.Ordinal);

        var canonicalContainers = CanonicalizeContainers(
            pageCandidates.Where(x => eligibleIds.Contains(x.CandidateId)).ToList());

        var acceptedObservationIds = lineDecisions
            .Where(x => x.Accepted)
            .Select(x => x.ObservationId)
            .ToHashSet(StringComparer.Ordinal);

        var acceptedObservations = containerObservations
            .Where(x => acceptedObservationIds.Contains(x.ObservationId))
            .ToList();

        var canonicalLines = rawLines
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .ThenByDescending(x => x.Confidence)
            .Select((line, index) => new CanonicalLine($"L{index + 1:0000}", line))
            .ToList();

        var ownership = new List<LineOwnershipDecision>(canonicalLines.Count);
        var assigned = canonicalContainers.ToDictionary(
            x => x,
            _ => new List<CanonicalLine>());
        var orphans = new List<CanonicalLine>();

        foreach (var line in canonicalLines)
        {
            token.ThrowIfCancellationRequested();

            var candidates = canonicalContainers
                .Select(container => EvaluateOwnership(
                    line.Line,
                    container,
                    acceptedObservations))
                .Where(x => x.Coverage >= 0.52)
                .OrderByDescending(x => x.Score)
                .ThenByDescending(x => x.Coverage)
                .ToList();

            if (candidates.Count == 0)
            {
                ownership.Add(new(
                    line.LineId,
                    null,
                    false,
                    "no_container_owner",
                    0,
                    0));
                orphans.Add(line);
                continue;
            }

            var best = candidates[0];
            var second = candidates.Count > 1 ? candidates[1] : null;

            bool ambiguous = second is not null &&
                             best.Score - second.Score < 0.75 &&
                             best.HasContainerEvidence == second.HasContainerEvidence;

            if (ambiguous)
            {
                ownership.Add(new(
                    line.LineId,
                    null,
                    false,
                    "ambiguous_container_owner",
                    best.Coverage,
                    best.Score));
                orphans.Add(line);
                continue;
            }

            assigned[best.Container].Add(line);
            ownership.Add(new(
                line.LineId,
                best.Container.Candidate.CandidateId,
                true,
                best.HasContainerEvidence
                    ? "container_ocr_evidence"
                    : "geometry_owner",
                best.Coverage,
                best.Score));
        }

        var drafts = new List<UnitDraft>();

        foreach (var container in canonicalContainers)
        {
            token.ThrowIfCancellationRequested();

            var lines = DeduplicateLines(assigned[container]);
            if (lines.Count == 0)
                continue;

            drafts.Add(new(
                ToBlock(-1, lines.Select(x => x.Line).ToList()),
                container.Candidate.CandidateId,
                lines.Select(x => x.LineId).ToList(),
                false,
                "container_owned"));
        }

        var orphanBlocks = orphanGrouper.Group(orphans.Select(x => x.Line).ToList());
        foreach (var block in orphanBlocks)
        {
            token.ThrowIfCancellationRequested();

            var lineIds = block.Lines
                .Select(line => orphans.FirstOrDefault(x => Equals(x.Line, line))?.LineId)
                .Where(id => !string.IsNullOrWhiteSpace(id))
                .Cast<string>()
                .Distinct(StringComparer.Ordinal)
                .ToList();

            drafts.Add(new(
                block,
                null,
                lineIds,
                true,
                "orphan_legacy_group"));
        }

        var orderedDrafts = drafts
            .OrderBy(x => x.Block.Y)
            .ThenBy(x => x.Block.X)
            .ToList();

        var units = orderedDrafts
            .Select((x, id) => x.Block with { Id = id })
            .ToList();

        var unitOwnership = orderedDrafts
            .Select((x, id) => new UnitOwnershipDecision(
                id,
                x.CandidateId,
                x.LineIds,
                x.IsOrphan,
                x.Reason))
            .ToList();

        return new OcrUnitBuildResult(
            units,
            canonicalContainers.Count,
            ownership.Count(x => x.Assigned),
            orphanBlocks.Count)
        {
            LineOwnership = ownership,
            UnitOwnership = unitOwnership
        };
    }

    static OwnershipCandidate EvaluateOwnership(
        OcrLine line,
        CanonicalContainer container,
        IReadOnlyList<OcrObservation> acceptedObservations)
    {
        double coverage = LineMaskCoverage(line, container.Candidate);
        bool hasEvidence = acceptedObservations.Any(observation =>
            container.SourceIds.Contains(observation.SourceKey) &&
            SameEvidenceLine(line, observation));

        double score =
            coverage * 10.0 +
            Math.Clamp(container.Candidate.Score, 0, 30) * 0.12 +
            Math.Clamp(container.Candidate.FillRatio, 0, 1) * 0.6 -
            CenterDistancePenalty(line, container.Candidate.Bounds) +
            (hasEvidence ? 3.0 : 0);

        return new(
            container,
            coverage,
            score,
            hasEvidence);
    }

    static List<CanonicalContainer> CanonicalizeContainers(
        IReadOnlyList<ContainerCandidate> candidates)
    {
        var ordered = candidates
            .OrderByDescending(x => x.Score)
            .ThenByDescending(x => x.FillRatio)
            .ThenByDescending(x => x.Bounds.Width * (long)x.Bounds.Height)
            .ToList();

        var result = new List<CanonicalContainer>();

        foreach (var candidate in ordered)
        {
            var existing = result.FirstOrDefault(x =>
                IsSamePhysicalContainer(x.Candidate.Bounds, candidate.Bounds));

            if (existing is null)
                result.Add(new CanonicalContainer(candidate));
            else
                existing.SourceIds.Add(candidate.CandidateId);
        }

        return result;
    }

    static bool IsSamePhysicalContainer(Rect a, Rect b)
    {
        double intersection = IntersectionArea(a, b);
        if (intersection <= 0)
            return false;

        double areaA = Math.Max(1.0, a.Width * (double)a.Height);
        double areaB = Math.Max(1.0, b.Width * (double)b.Height);
        double containment = intersection / Math.Min(areaA, areaB);
        double union = areaA + areaB - intersection;
        double iou = union <= 0 ? 0 : intersection / union;

        return containment >= 0.82 || iou >= 0.60;
    }

    static double LineMaskCoverage(
        OcrLine line,
        ContainerCandidate candidate)
    {
        if (candidate.MaskWidth <= 0 ||
            candidate.MaskHeight <= 0 ||
            candidate.Mask.LongLength != (long)candidate.MaskWidth * candidate.MaskHeight)
            return 0;

        int inside = 0;
        const int samplesX = 5;
        const int samplesY = 3;
        int total = samplesX * samplesY;

        for (int iy = 0; iy < samplesY; iy++)
        {
            double ty = 0.25 + iy * 0.25;
            int y = (int)Math.Round(line.Y + line.H * ty);

            for (int ix = 0; ix < samplesX; ix++)
            {
                double tx = 0.10 + ix * 0.20;
                int x = (int)Math.Round(line.X + line.W * tx);
                int lx = x - candidate.Bounds.X;
                int ly = y - candidate.Bounds.Y;

                if (lx >= 0 &&
                    ly >= 0 &&
                    lx < candidate.MaskWidth &&
                    ly < candidate.MaskHeight &&
                    candidate.Mask[ly * candidate.MaskWidth + lx] != 0)
                {
                    inside++;
                }
            }
        }

        return inside / (double)Math.Max(1, total);
    }

    static double CenterDistancePenalty(
        OcrLine line,
        Rect bounds)
    {
        double cx = line.X + line.W / 2.0;
        double cy = line.Y + line.H / 2.0;
        double bx = bounds.X + bounds.Width / 2.0;
        double by = bounds.Y + bounds.Height / 2.0;
        double dx = (cx - bx) / Math.Max(20, bounds.Width);
        double dy = (cy - by) / Math.Max(20, bounds.Height);

        return Math.Sqrt(dx * dx + dy * dy);
    }

    static bool SameEvidenceLine(
        OcrLine line,
        OcrObservation observation)
    {
        if (!string.Equals(
                Normalize(line.Text),
                Normalize(observation.Text),
                StringComparison.OrdinalIgnoreCase))
            return false;

        double intersection = IntersectionArea(
            ToRect(line),
            new Rect(
                (int)Math.Floor(observation.X),
                (int)Math.Floor(observation.Y),
                Math.Max(1, (int)Math.Ceiling(observation.W)),
                Math.Max(1, (int)Math.Ceiling(observation.H))));

        double maxArea = Math.Max(
            Math.Max(1, line.W * line.H),
            Math.Max(1, observation.W * observation.H));

        return intersection / maxArea >= 0.50;
    }

    static List<CanonicalLine> DeduplicateLines(
        IReadOnlyList<CanonicalLine> input)
    {
        var ordered = input
            .OrderByDescending(x => x.Line.Confidence)
            .ToList();

        var kept = new List<CanonicalLine>();

        foreach (var candidate in ordered)
        {
            if (!kept.Any(x => IsSameLine(x.Line, candidate.Line)))
                kept.Add(candidate);
        }

        return OrderLines(kept);
    }

    static bool IsSameLine(
        OcrLine a,
        OcrLine b)
    {
        double intersection = IntersectionArea(ToRect(a), ToRect(b));
        double minArea = Math.Max(1.0, Math.Min(a.W * a.H, b.W * b.H));

        if (intersection / minArea >= 0.72)
            return true;

        string at = Normalize(a.Text);
        string bt = Normalize(b.Text);

        if (at.Length == 0 ||
            bt.Length == 0 ||
            !string.Equals(at, bt, StringComparison.OrdinalIgnoreCase))
            return false;

        double acx = a.X + a.W / 2.0;
        double acy = a.Y + a.H / 2.0;
        double bcx = b.X + b.W / 2.0;
        double bcy = b.Y + b.H / 2.0;
        double distance = Math.Sqrt(
            Math.Pow(acx - bcx, 2) +
            Math.Pow(acy - bcy, 2));

        return distance <= Math.Max(
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

    static List<OcrLine> OrderLines(IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();
        bool vertical =
            list.Count >= 2 &&
            list.Count(IsVertical) > list.Count / 2;

        return vertical
            ? list.OrderByDescending(x => x.X).ThenBy(x => x.Y).ToList()
            : list.OrderBy(x => x.Y).ThenBy(x => x.X).ToList();
    }

    static List<CanonicalLine> OrderLines(IEnumerable<CanonicalLine> lines)
    {
        var list = lines.ToList();
        bool vertical =
            list.Count >= 2 &&
            list.Count(x => IsVertical(x.Line)) > list.Count / 2;

        return vertical
            ? list.OrderByDescending(x => x.Line.X).ThenBy(x => x.Line.Y).ToList()
            : list.OrderBy(x => x.Line.Y).ThenBy(x => x.Line.X).ToList();
    }

    static bool IsVertical(OcrLine line)
        => line.H > line.W * 1.30;

    static Rect ToRect(OcrLine line)
        => new(
            (int)Math.Floor(line.X),
            (int)Math.Floor(line.Y),
            Math.Max(1, (int)Math.Ceiling(line.W)),
            Math.Max(1, (int)Math.Ceiling(line.H)));

    static double IntersectionArea(Rect a, Rect b)
    {
        int left = Math.Max(a.Left, b.Left);
        int top = Math.Max(a.Top, b.Top);
        int right = Math.Min(a.Right, b.Right);
        int bottom = Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               (double)Math.Max(0, bottom - top);
    }

    static string Normalize(string text)
        => new(
            text
                .Where(char.IsLetterOrDigit)
                .Select(char.ToLowerInvariant)
                .ToArray());
}
