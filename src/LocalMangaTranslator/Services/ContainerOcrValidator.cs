using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

/// <summary>Separate geometry eligibility and repeated OCR evidence. Neither approves deletion.</summary>
public static class ContainerOcrValidator
{
    public static ContainerOcrDecision ValidateCandidate(ContainerCandidate c)
    {
        string reason = "eligible_for_ocr";
        if (c.Bounds.X < 0 || c.Bounds.Y < 0 || c.Bounds.Width < 24 || c.Bounds.Height < 18 ||
            c.MaskWidth != c.Bounds.Width || c.MaskHeight != c.Bounds.Height ||
            c.Mask.LongLength != (long)c.MaskWidth * c.MaskHeight)
            reason = "invalid_geometry";
        else if (!double.IsFinite(c.Score) || !double.IsFinite(c.FillRatio) ||
                 c.Score < 4.0 || c.FillRatio < 0.55 || c.FillRatio > 1 || c.BorderTouches != 0)
            reason = "weak_or_border_geometry";
        else if (c.Mask.Count(v => v != 0) < c.Mask.Length * 0.35)
            reason = "insufficient_interior";
        return new(c.CandidateId, reason == "eligible_for_ocr", reason);
    }

    public static IReadOnlyList<ContainerLineDecision> ValidateLines(
        IReadOnlyList<ContainerCandidate> candidates,
        IReadOnlyList<OcrObservation> observations,
        CancellationToken token = default)
    {
        var byId = candidates.ToDictionary(x => x.CandidateId);
        var eligible = candidates.ToDictionary(x => x.CandidateId, x => ValidateCandidate(x).Eligible);
        var coverageById = observations.ToDictionary(x => x.ObservationId,
            x => byId.TryGetValue(x.SourceKey, out var c) ? MaskCoverage(c, x) : 0);
        var result = new List<ContainerLineDecision>();
        foreach (var o in observations)
        {
            token.ThrowIfCancellationRequested();
            byId.TryGetValue(o.SourceKey, out var candidate);
            double coverage = coverageById[o.ObservationId];
            string reason = candidate is null || !eligible[candidate.CandidateId]
                ? "ineligible_candidate"
                : !Usable(o) ? "weak_text_evidence"
                : coverage < 0.90 ? "outside_candidate_interior"
                : !observations.Any(other =>
                    other.SourceKey == o.SourceKey && ComplementaryPass(o.Pass, other.Pass) &&
                    Usable(other) && Normalize(o.Text) == Normalize(other.Text) &&
                    coverageById[other.ObservationId] >= 0.90 && SameLine(o, other))
                    ? "no_cross_scale_agreement" : "cross_scale_agreement";
            result.Add(new(o.ObservationId, o.SourceKey, reason == "cross_scale_agreement", reason, coverage));
        }
        return result;
    }

    static bool ComplementaryPass(OcrPassKind a, OcrPassKind b)
        => (a == OcrPassKind.Container1x && b == OcrPassKind.Container2x) ||
           (a == OcrPassKind.Container2x && b == OcrPassKind.Container1x);

    static bool Usable(OcrObservation o)
        => double.IsFinite(o.X) && double.IsFinite(o.Y) && double.IsFinite(o.W) && double.IsFinite(o.H) &&
           o.W >= 3 && o.H >= 3 && float.IsFinite(o.Confidence) && o.Confidence >= 0.65f &&
           Normalize(o.Text).Length >= 2;

    static string Normalize(string text)
        => new string(text.Where(char.IsLetterOrDigit).ToArray()).ToUpperInvariant();

    static bool SameLine(OcrObservation a, OcrObservation b)
    {
        double intersection = Math.Max(0, Math.Min(a.X + a.W, b.X + b.W) - Math.Max(a.X, b.X)) *
                              Math.Max(0, Math.Min(a.Y + a.H, b.Y + b.H) - Math.Max(a.Y, b.Y));
        // Both boxes must agree; a tiny fragment inside a large box is insufficient.
        return intersection / Math.Max(a.W * a.H, b.W * b.H) >= 0.65;
    }

    public static double MaskCoverage(ContainerCandidate c, OcrObservation o)
    {
        if (!Usable(o) || c.MaskWidth <= 0 || c.MaskHeight <= 0 ||
            c.Mask.LongLength != (long)c.MaskWidth * c.MaskHeight)
            return 0;
        double left = Math.Floor(o.X - c.Bounds.X), top = Math.Floor(o.Y - c.Bounds.Y);
        double right = Math.Ceiling(o.X + o.W - c.Bounds.X), bottom = Math.Ceiling(o.Y + o.H - c.Bounds.Y);
        double area = (right - left) * (bottom - top);
        if (!double.IsFinite(area) || area <= 0) return 0;
        int x0 = (int)Math.Clamp(left, 0, c.MaskWidth), y0 = (int)Math.Clamp(top, 0, c.MaskHeight);
        int x1 = (int)Math.Clamp(right, 0, c.MaskWidth), y1 = (int)Math.Clamp(bottom, 0, c.MaskHeight);
        long inside = 0;
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                if (c.Mask[y * c.MaskWidth + x] != 0) inside++;
        return inside / area;
    }
}
