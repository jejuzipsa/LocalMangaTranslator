using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// 0009 conservative container-first OCR unit builder with a narrow rescue pass.
///
/// A page candidate may own a line only when container OCR evidence supports it
/// or when geometry is strong enough to be unambiguous. Broad/weak candidate
/// overlap is diagnostic evidence, not automatic ownership.
///
/// Owned lines are clustered again inside the container before translation-unit
/// creation. Ambiguous/weak ownership lines stay isolated so the legacy grouper
/// cannot silently re-merge them into a bad unit.
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
        bool HasContainerEvidence,
        double CenterPenalty);

    sealed record HeldLine(
        CanonicalLine Line,
        string Reason);

    sealed record UnitDraft(
        OcrTextBlock Block,
        string? CandidateId,
        IReadOnlyList<string> LineIds,
        bool IsOrphan,
        string Reason);

    readonly OcrBlockGrouper orphanGrouper = new();
    readonly OcrBlockGrouper containerGrouper = new();

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
            pageCandidates
                .Where(x => eligibleIds.Contains(x.CandidateId))
                .ToList());

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

        // Truly unowned text can still use the old local grouper.
        var legacyOrphans = new List<CanonicalLine>();

        // Lines that touched a candidate but failed confidence/ambiguity gates
        // must not be re-merged by the legacy grouper.
        var isolatedHeldLines = new List<HeldLine>();

        foreach (var line in canonicalLines)
        {
            token.ThrowIfCancellationRequested();

            var evaluated = canonicalContainers
                .Select(container => EvaluateOwnership(
                    line.Line,
                    container,
                    acceptedObservations))
                .Where(x => x.Coverage >= 0.35)
                .OrderByDescending(x => x.Score)
                .ThenByDescending(x => x.Coverage)
                .ToList();

            var candidates = evaluated
                .Where(x => IsTrustworthyOwner(line.Line, x))
                .ToList();

            if (candidates.Count == 0)
            {
                bool touchedCandidate = evaluated.Count > 0;
                string reason = touchedCandidate
                    ? "weak_container_owner"
                    : "no_container_owner";

                ownership.Add(new(
                    line.LineId,
                    null,
                    false,
                    reason,
                    evaluated.FirstOrDefault()?.Coverage ?? 0,
                    evaluated.FirstOrDefault()?.Score ?? 0));

                if (touchedCandidate)
                    isolatedHeldLines.Add(new(line, reason));
                else
                    legacyOrphans.Add(line);

                continue;
            }

            var best = candidates[0];
            var second = candidates.Count > 1 ? candidates[1] : null;

            if (second is not null && IsAmbiguous(best, second))
            {
                ownership.Add(new(
                    line.LineId,
                    null,
                    false,
                    "ambiguous_container_owner",
                    best.Coverage,
                    best.Score));

                isolatedHeldLines.Add(new(
                    line,
                    "ambiguous_container_owner"));

                continue;
            }

            assigned[best.Container].Add(line);

            ownership.Add(new(
                line.LineId,
                best.Container.Candidate.CandidateId,
                true,
                best.HasContainerEvidence
                    ? "container_ocr_evidence"
                    : "strong_geometry_owner",
                best.Coverage,
                best.Score));
        }

        RescueHeldLines(
            isolatedHeldLines,
            assigned,
            ownership,
            canonicalContainers,
            acceptedObservations,
            token);

        var drafts = new List<UnitDraft>();

        foreach (var container in canonicalContainers)
        {
            token.ThrowIfCancellationRequested();

            var lines = DeduplicateLines(assigned[container]);
            if (lines.Count == 0)
                continue;

            // A broad geometric candidate is not proof that all of its text is
            // one utterance. Reuse the local grouping rules inside the physical
            // container so distant clusters cannot become one huge unit.
            var clusters = containerGrouper.Group(
                lines.Select(x => x.Line).ToList());

            bool split = clusters.Count > 1;

            foreach (var cluster in clusters)
            {
                token.ThrowIfCancellationRequested();

                var clusterLines = MatchCanonicalLines(
                    lines,
                    cluster.Lines);

                if (clusterLines.Count == 0)
                    continue;

                drafts.Add(new(
                    ToBlock(
                        -1,
                        clusterLines.Select(x => x.Line).ToList()),
                    container.Candidate.CandidateId,
                    clusterLines.Select(x => x.LineId).ToList(),
                    false,
                    split
                        ? "container_clustered"
                        : "container_owned"));
            }
        }

        var orphanBlocks = orphanGrouper.Group(
            legacyOrphans.Select(x => x.Line).ToList());

        foreach (var block in orphanBlocks)
        {
            token.ThrowIfCancellationRequested();

            var blockLines = MatchCanonicalLines(
                legacyOrphans,
                block.Lines);

            if (blockLines.Count == 0)
                continue;

            drafts.Add(new(
                ToBlock(
                    -1,
                    blockLines.Select(x => x.Line).ToList()),
                null,
                blockLines.Select(x => x.LineId).ToList(),
                true,
                "orphan_legacy_group"));
        }

        foreach (var held in isolatedHeldLines)
        {
            token.ThrowIfCancellationRequested();

            drafts.Add(new(
                ToBlock(-1, [held.Line.Line]),
                null,
                [held.Line.LineId],
                true,
                held.Reason == "ambiguous_container_owner"
                    ? "orphan_isolated_ambiguous"
                    : "orphan_isolated_weak_owner"));
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

        int orphanUnitCount = orderedDrafts.Count(x => x.IsOrphan);

        return new OcrUnitBuildResult(
            units,
            canonicalContainers.Count,
            ownership.Count(x => x.Assigned),
            orphanUnitCount)
        {
            LineOwnership = ownership,
            UnitOwnership = unitOwnership
        };
    }

    static void RescueHeldLines(
        List<HeldLine> heldLines,
        Dictionary<CanonicalContainer, List<CanonicalLine>> assigned,
        List<LineOwnershipDecision> ownership,
        IReadOnlyList<CanonicalContainer> containers,
        IReadOnlyList<OcrObservation> acceptedObservations,
        CancellationToken token)
    {
        // 0008 correctly stopped half-overlapping artwork from becoming a
        // container owner, but it also isolated an occasional real line inside
        // an otherwise coherent speech balloon. Rescue only weak (not
        // ambiguous) lines when a strong neighboring line already owns the same
        // physical container. This does not lower the global ownership gate.
        for (int i = heldLines.Count - 1; i >= 0; i--)
        {
            token.ThrowIfCancellationRequested();

            var held = heldLines[i];
            if (!string.Equals(
                    held.Reason,
                    "weak_container_owner",
                    StringComparison.Ordinal))
                continue;

            var line = held.Line.Line;
            if (line.Confidence < 0.72f ||
                MeaningfulTextLength(line.Text) < 2)
                continue;

            var candidates = containers
                .Select(container => EvaluateOwnership(
                    line,
                    container,
                    acceptedObservations))
                .Where(x =>
                    x.Coverage >= 0.64 &&
                    x.Container.Candidate.Score >= 4.60 &&
                    x.Container.Candidate.FillRatio >= 0.60 &&
                    x.CenterPenalty <= 0.95)
                .OrderByDescending(x => x.Score)
                .ThenByDescending(x => x.Coverage)
                .ToList();

            if (candidates.Count == 0)
                continue;

            var best = candidates[0];
            var second = candidates.Count > 1
                ? candidates[1]
                : null;

            // Rescue must be more decisive than ordinary assignment because it
            // exists specifically to avoid restoring artwork false positives.
            if (second is not null &&
                (best.Score - second.Score < 1.45 ||
                 best.Coverage - second.Coverage < 0.10))
                continue;

            if (!assigned.TryGetValue(
                    best.Container,
                    out var neighbors) ||
                neighbors.Count == 0 ||
                !neighbors.Any(x => IsCompatibleNeighbor(
                    line,
                    x.Line,
                    best.Container.Candidate.Bounds)))
                continue;

            neighbors.Add(held.Line);

            int ownershipIndex = ownership.FindIndex(x =>
                string.Equals(
                    x.LineId,
                    held.Line.LineId,
                    StringComparison.Ordinal));

            if (ownershipIndex >= 0)
            {
                ownership[ownershipIndex] = new LineOwnershipDecision(
                    held.Line.LineId,
                    best.Container.Candidate.CandidateId,
                    true,
                    best.HasContainerEvidence
                        ? "rescued_container_ocr_evidence"
                        : "rescued_neighbor_geometry",
                    best.Coverage,
                    best.Score);
            }

            heldLines.RemoveAt(i);
        }

        // A real speech balloon can occasionally have no single line strong
        // enough to seed ownership after 0008's stricter candidate-quality
        // gates. Two or more high-confidence text lines that independently
        // prefer the same candidate and form a coherent local stack are much
        // stronger evidence than one isolated artwork fragment.
        var peerCandidates = new List<(HeldLine Held, OwnershipCandidate Best)>();

        foreach (var held in heldLines)
        {
            token.ThrowIfCancellationRequested();

            if (!string.Equals(
                    held.Reason,
                    "weak_container_owner",
                    StringComparison.Ordinal))
                continue;

            var line = held.Line.Line;
            if (line.Confidence < 0.78f ||
                MeaningfulTextLength(line.Text) < 2)
                continue;

            var evaluated = containers
                .Select(container => EvaluateOwnership(
                    line,
                    container,
                    acceptedObservations))
                .Where(x =>
                    x.Coverage >= 0.80 &&
                    x.Container.Candidate.Score >= 4.00 &&
                    x.Container.Candidate.FillRatio >= 0.45 &&
                    x.CenterPenalty <= 1.05)
                .OrderByDescending(x => x.Score)
                .ThenByDescending(x => x.Coverage)
                .ToList();

            if (evaluated.Count == 0)
                continue;

            var best = evaluated[0];
            var second = evaluated.Count > 1
                ? evaluated[1]
                : null;

            if (second is not null &&
                (best.Score - second.Score < 0.75 ||
                 best.Coverage - second.Coverage < 0.06))
                continue;

            peerCandidates.Add((held, best));
        }

        var peerRescued = new HashSet<string>(
            StringComparer.Ordinal);

        foreach (var group in peerCandidates.GroupBy(x => x.Best.Container))
        {
            var items = group.ToList();
            if (items.Count < 2)
                continue;

            var coherent = items
                .Where(item => items.Any(other =>
                    !ReferenceEquals(
                        item.Held,
                        other.Held) &&
                    IsCompatibleNeighbor(
                        item.Held.Line.Line,
                        other.Held.Line.Line,
                        item.Best.Container.Candidate.Bounds)))
                .ToList();

            if (coherent.Count < 2)
                continue;

            foreach (var item in coherent)
            {
                if (!peerRescued.Add(
                        item.Held.Line.LineId))
                    continue;

                assigned[item.Best.Container].Add(
                    item.Held.Line);

                int ownershipIndex = ownership.FindIndex(x =>
                    string.Equals(
                        x.LineId,
                        item.Held.Line.LineId,
                        StringComparison.Ordinal));

                if (ownershipIndex >= 0)
                {
                    ownership[ownershipIndex] = new LineOwnershipDecision(
                        item.Held.Line.LineId,
                        item.Best.Container.Candidate.CandidateId,
                        true,
                        "rescued_peer_cluster",
                        item.Best.Coverage,
                        item.Best.Score);
                }
            }
        }

        if (peerRescued.Count > 0)
        {
            heldLines.RemoveAll(x =>
                peerRescued.Contains(
                    x.Line.LineId));
        }
    }

    static bool IsCompatibleNeighbor(
        OcrLine candidate,
        OcrLine neighbor,
        Rect container)
    {
        bool candidateVertical = IsVertical(candidate);
        bool neighborVertical = IsVertical(neighbor);

        if (candidateVertical != neighborVertical)
            return false;

        double candidateSize = Math.Max(1, Math.Min(candidate.W, candidate.H));
        double neighborSize = Math.Max(1, Math.Min(neighbor.W, neighbor.H));
        double sizeRatio = candidateSize / neighborSize;
        if (sizeRatio < 0.55 || sizeRatio > 1.85)
            return false;

        double ccx = candidate.X + candidate.W / 2.0;
        double ccy = candidate.Y + candidate.H / 2.0;
        double ncx = neighbor.X + neighbor.W / 2.0;
        double ncy = neighbor.Y + neighbor.H / 2.0;

        if (candidateVertical)
        {
            double horizontalGap = Math.Abs(ccx - ncx);
            double verticalGap = Math.Abs(ccy - ncy);

            return horizontalGap <= Math.Max(12, Math.Max(candidate.W, neighbor.W) * 1.90) &&
                   verticalGap <= Math.Max(24, container.Height * 0.60);
        }

        double rowGap = Math.Abs(ccy - ncy);
        double columnGap = Math.Abs(ccx - ncx);

        return rowGap <= Math.Max(12, Math.Max(candidate.H, neighbor.H) * 2.15) &&
               columnGap <= Math.Max(30, container.Width * 0.48);
    }

    static int MeaningfulTextLength(string text)
        => text.Count(char.IsLetterOrDigit);

    static OwnershipCandidate EvaluateOwnership(
        OcrLine line,
        CanonicalContainer container,
        IReadOnlyList<OcrObservation> acceptedObservations)
    {
        double coverage = LineMaskCoverage(
            line,
            container.Candidate);

        bool hasEvidence = acceptedObservations.Any(observation =>
            container.SourceIds.Contains(observation.SourceKey) &&
            SameEvidenceLine(line, observation));

        double centerPenalty = CenterDistancePenalty(
            line,
            container.Candidate.Bounds);

        double score =
            coverage * 10.0 +
            Math.Clamp(container.Candidate.Score, 0, 30) * 0.12 +
            Math.Clamp(container.Candidate.FillRatio, 0, 1) * 0.6 -
            centerPenalty +
            (hasEvidence ? 3.0 : 0);

        return new(
            container,
            coverage,
            score,
            hasEvidence,
            centerPenalty);
    }

    static bool IsTrustworthyOwner(
        OcrLine line,
        OwnershipCandidate candidate)
    {
        if (!double.IsFinite(candidate.Coverage) ||
            !double.IsFinite(candidate.Score) ||
            !double.IsFinite(candidate.CenterPenalty))
            return false;

        // A validated 1x/2x container OCR pair is strong evidence, but the
        // merged line must still be substantially inside the same safe mask.
        if (candidate.HasContainerEvidence)
            return candidate.Coverage >= 0.72;

        // Geometry-only ownership is intentionally much stricter than 0007.
        // The old 0.52 threshold admitted half-overlapping lines and produced
        // broad mixed units in the 0006 regression pages.
        return candidate.Coverage >= 0.90 &&
               line.Confidence >= 0.60f &&
               candidate.Container.Candidate.Score >= 4.60 &&
               candidate.Container.Candidate.FillRatio >= 0.60 &&
               candidate.CenterPenalty <= 0.95;
    }

    static bool IsAmbiguous(
        OwnershipCandidate best,
        OwnershipCandidate second)
    {
        // Strong OCR evidence may break a geometry-only tie.
        if (best.HasContainerEvidence &&
            !second.HasContainerEvidence &&
            best.Score - second.Score >= 0.45)
            return false;

        double scoreGap = best.Score - second.Score;
        double coverageGap = best.Coverage - second.Coverage;

        return scoreGap < 1.25 ||
               (!best.HasContainerEvidence &&
                !second.HasContainerEvidence &&
                coverageGap < 0.08);
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
                IsSamePhysicalContainer(
                    x.Candidate.Bounds,
                    candidate.Bounds));

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
            candidate.Mask.LongLength !=
                (long)candidate.MaskWidth * candidate.MaskHeight)
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
        // Confidence-only ordering used to keep a tiny high-confidence
        // fragment such as "THE MIDDLE" and discard the wider
        // "YOU'RE STILL IN THE MIDDLE" observation at the same location.
        // Keep the geometrically/textually more complete representative when
        // confidence is still comparable.
        var ordered = input
            .OrderByDescending(x => x.Line.Confidence)
            .ToList();

        var kept = new List<CanonicalLine>();

        foreach (var candidate in ordered)
        {
            int duplicateIndex =
                kept.FindIndex(x =>
                    IsSameLine(
                        x.Line,
                        candidate.Line));

            if (duplicateIndex < 0)
            {
                kept.Add(
                    candidate);

                continue;
            }

            if (PreferMoreCompleteLine(
                    candidate.Line,
                    kept[duplicateIndex].Line))
            {
                kept[duplicateIndex] =
                    candidate;
            }
        }

        return OrderLines(
            kept);
    }

    static bool PreferMoreCompleteLine(
        OcrLine candidate,
        OcrLine current)
    {
        string candidateText =
            Normalize(
                candidate.Text);

        string currentText =
            Normalize(
                current.Text);

        int candidateLength =
            candidateText.Length;

        int currentLength =
            currentText.Length;

        double candidateArea =
            Math.Max(
                1.0,
                candidate.W *
                candidate.H);

        double currentArea =
            Math.Max(
                1.0,
                current.W *
                current.H);

        bool textContainsCurrent =
            currentLength > 0 &&
            candidateLength >
                currentLength &&
            candidateText.Contains(
                currentText,
                StringComparison.OrdinalIgnoreCase);

        bool clearlyMoreComplete =
            candidateLength >=
                currentLength + 3 &&
            candidateArea >=
                currentArea * 1.20;

        bool confidenceComparable =
            candidate.Confidence >=
                current.Confidence -
                0.08f;

        return confidenceComparable &&
               (textContainsCurrent ||
                clearlyMoreComplete);
    }

    static List<CanonicalLine> MatchCanonicalLines(
        IReadOnlyList<CanonicalLine> source,
        IReadOnlyList<OcrLine> lines)
    {
        var remaining = source.ToList();
        var result = new List<CanonicalLine>();

        foreach (var line in lines)
        {
            int index = remaining.FindIndex(x =>
                ReferenceEquals(x.Line, line) ||
                Equals(x.Line, line));

            if (index < 0)
                continue;

            result.Add(remaining[index]);
            remaining.RemoveAt(index);
        }

        return OrderLines(result);
    }

    static bool IsSameLine(
        OcrLine a,
        OcrLine b)
    {
        double intersection = IntersectionArea(
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

    static List<OcrLine> OrderLines(
        IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();

        bool vertical =
            list.Count >= 2 &&
            list.Count(IsVertical) > list.Count / 2;

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

    static List<CanonicalLine> OrderLines(
        IEnumerable<CanonicalLine> lines)
    {
        var list = lines.ToList();

        bool vertical =
            list.Count >= 2 &&
            list.Count(x => IsVertical(x.Line)) >
            list.Count / 2;

        return vertical
            ? list
                .OrderByDescending(x => x.Line.X)
                .ThenBy(x => x.Line.Y)
                .ToList()
            : list
                .OrderBy(x => x.Line.Y)
                .ThenBy(x => x.Line.X)
                .ToList();
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
