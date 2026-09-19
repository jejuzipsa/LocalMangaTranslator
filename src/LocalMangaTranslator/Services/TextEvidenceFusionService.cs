using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Fuses OCR evidence from the independent RT-DETR text-region branch with
/// page-wide OCR. Region detection never supplies the recognized string; it
/// only contributes spatial evidence.
/// </summary>
public static class TextEvidenceFusionService
{
    public static IReadOnlyList<RegionLineDecision> Validate(
        IReadOnlyList<PageRegion> regions,
        IReadOnlyList<OcrObservation> regionObservations,
        IReadOnlyList<OcrObservation> globalObservations,
        CancellationToken token = default)
    {
        var byId =
            regions.ToDictionary(
                x => x.RegionId,
                StringComparer.Ordinal);

        var decisions =
            new List<RegionLineDecision>(
                regionObservations.Count);

        foreach (var observation in regionObservations)
        {
            token.ThrowIfCancellationRequested();

            if (!byId.TryGetValue(
                    observation.SourceKey,
                    out var region) ||
                region.Kind is not
                    (PageRegionKind.TextBubble or
                     PageRegionKind.TextFree))
            {
                decisions.Add(
                    new RegionLineDecision(
                        observation.ObservationId,
                        observation.SourceKey,
                        false,
                        "missing_text_region"));

                continue;
            }

            string normalized =
                Normalize(
                    observation.Text);

            if (!Usable(
                    observation,
                    normalized))
            {
                decisions.Add(
                    new RegionLineDecision(
                        observation.ObservationId,
                        observation.SourceKey,
                        false,
                        "weak_ocr"));

                continue;
            }

            bool crossScale =
                regionObservations.Any(
                    other =>
                        !ReferenceEquals(
                            other,
                            observation) &&
                        string.Equals(
                            other.SourceKey,
                            observation.SourceKey,
                            StringComparison.Ordinal) &&
                        ComplementaryPass(
                            observation.Pass,
                            other.Pass) &&
                        Normalize(
                            other.Text) ==
                        normalized &&
                        SameLine(
                            observation,
                            other));

            bool globalAgreement =
                globalObservations.Any(
                    other =>
                        Normalize(
                            other.Text) ==
                        normalized &&
                        SameLine(
                            observation,
                            other));

            bool strongRegionObservation =
                region.Score >= 0.62f &&
                observation.Confidence >= 0.78f &&
                normalized.Length >= 2;

            string reason =
                crossScale
                    ? "region_cross_scale"
                    : globalAgreement
                        ? "region_global_agreement"
                        : strongRegionObservation
                            ? "strong_region_ocr"
                            : "uncorroborated_region_ocr";

            decisions.Add(
                new RegionLineDecision(
                    observation.ObservationId,
                    observation.SourceKey,
                    reason !=
                        "uncorroborated_region_ocr",
                    reason));
        }

        return decisions;
    }

    static bool ComplementaryPass(
        OcrPassKind a,
        OcrPassKind b)
        => (a == OcrPassKind.Region1x &&
            b == OcrPassKind.Region2x) ||
           (a == OcrPassKind.Region2x &&
            b == OcrPassKind.Region1x);

    static bool Usable(
        OcrObservation observation,
        string normalized)
        => double.IsFinite(
               observation.X) &&
           double.IsFinite(
               observation.Y) &&
           double.IsFinite(
               observation.W) &&
           double.IsFinite(
               observation.H) &&
           observation.W >= 2 &&
           observation.H >= 2 &&
           float.IsFinite(
               observation.Confidence) &&
           observation.Confidence >= 0.45f &&
           normalized.Length >= 1;

    static bool SameLine(
        OcrObservation a,
        OcrObservation b)
    {
        double left =
            Math.Max(
                a.X,
                b.X);

        double top =
            Math.Max(
                a.Y,
                b.Y);

        double right =
            Math.Min(
                a.X + a.W,
                b.X + b.W);

        double bottom =
            Math.Min(
                a.Y + a.H,
                b.Y + b.H);

        double intersection =
            Math.Max(
                0,
                right - left) *
            Math.Max(
                0,
                bottom - top);

        if (intersection <= 0)
            return false;

        double areaA =
            Math.Max(
                1,
                a.W *
                a.H);

        double areaB =
            Math.Max(
                1,
                b.W *
                b.H);

        return intersection /
               Math.Max(
                   areaA,
                   areaB) >=
               0.42;
    }

    static string Normalize(
        string text)
        => new(
            text
                .Where(
                    char.IsLetterOrDigit)
                .Select(
                    char.ToUpperInvariant)
                .ToArray());
}
