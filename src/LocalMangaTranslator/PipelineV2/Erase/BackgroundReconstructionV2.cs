using LocalMangaTranslator.PipelineV2.Detection;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2BackgroundReconstructionAudit(
    string TextRegionId,
    string Kind,
    Rect Bounds,
    string Strategy,
    int MaskPixels,
    int SampleCount,
    int BackgroundB,
    int BackgroundG,
    int BackgroundR,
    double DominantMatchRatio,
    double P75ColorDistance,
    double P90ColorDistance,
    bool FlatAccepted,
    string Reason);

/// <summary>
/// Chooses how erased comic lettering is reconstructed.
///
/// A text mask describes foreground ink. It does not describe what should
/// replace that ink. For locally flat Bubble/Caption backgrounds we restore
/// the robust local background color directly. Complex artwork remains on the
/// ordinary inpaint path.
///
/// Sampling deliberately uses pixels INSIDE the immutable detector text box
/// but OUTSIDE the accepted glyph mask. This is important for large display
/// lettering whose TextBubble nearly fills its parent Bubble: an outside ring
/// would mostly see the balloon border instead of the background behind text.
/// </summary>
public static class BackgroundReconstructionV2
{
    const double MatchDistance = 28.0;

    public static V2BackgroundReconstructionAudit Analyze(
        Mat source,
        V2TextTarget target,
        Mat glyphMask)
    {
        int maskPixels =
            Cv2.CountNonZero(
                glyphMask);

        if (maskPixels < 2)
        {
            return EmptyAudit(
                target,
                maskPixels,
                "TELEA",
                "mask_empty");
        }

        Rect bounds =
            ClampRect(
                target.TextBounds,
                source.Cols,
                source.Rows);

        if (bounds.Width <= 4 ||
            bounds.Height <= 4)
        {
            return EmptyAudit(
                target,
                maskPixels,
                "TELEA",
                "bounds_too_small");
        }

        int inset =
            Math.Clamp(
                (int)Math.Round(
                    Math.Min(
                        bounds.Width,
                        bounds.Height) *
                    0.03),
                2,
                8);

        var sampleBounds =
            new Rect(
                bounds.X + inset,
                bounds.Y + inset,
                Math.Max(
                    0,
                    bounds.Width - inset * 2),
                Math.Max(
                    0,
                    bounds.Height - inset * 2));

        if (sampleBounds.Width <= 2 ||
            sampleBounds.Height <= 2)
        {
            sampleBounds =
                bounds;
        }

        var samples =
            new List<Vec3b>();

        for (int y =
                 sampleBounds.Top;
             y <
             sampleBounds.Bottom;
             y += 2)
        {
            for (int x =
                     sampleBounds.Left;
                 x <
                 sampleBounds.Right;
                 x += 2)
            {
                if (glyphMask.At<byte>(
                        y,
                        x) != 0)
                {
                    continue;
                }

                samples.Add(
                    source.At<Vec3b>(
                        y,
                        x));
            }
        }

        int minimumSamples =
            Math.Max(
                32,
                (sampleBounds.Width *
                 sampleBounds.Height) /
                160);

        if (samples.Count <
            minimumSamples)
        {
            return EmptyAudit(
                target,
                maskPixels,
                "TELEA",
                "insufficient_background_samples",
                samples.Count);
        }

        var blues =
            samples
                .Select(x =>
                    x.Item0)
                .OrderBy(x =>
                    x)
                .ToArray();

        var greens =
            samples
                .Select(x =>
                    x.Item1)
                .OrderBy(x =>
                    x)
                .ToArray();

        var reds =
            samples
                .Select(x =>
                    x.Item2)
                .OrderBy(x =>
                    x)
                .ToArray();

        int medianIndex =
            samples.Count /
            2;

        var background =
            new Vec3b(
                blues[medianIndex],
                greens[medianIndex],
                reds[medianIndex]);

        var distances =
            samples
                .Select(x =>
                    ColorDistance(
                        x,
                        background))
                .OrderBy(x =>
                    x)
                .ToArray();

        double p75 =
            Percentile(
                distances,
                0.75);

        double p90 =
            Percentile(
                distances,
                0.90);

        double dominantRatio =
            distances.Count(x =>
                x <=
                MatchDistance) /
            (double)Math.Max(
                1,
                distances.Length);

        // Conservative flat decision:
        // - at least roughly 70% of the visible, non-glyph pixels belong to
        //   one local color cluster;
        // - the central 75% is genuinely tight;
        // - the 90th percentile may contain anti-aliasing, balloon borders,
        //   or a little artwork, but must not be wildly different.
        //
        // Ambiguous targets always fall back to Telea.
        bool flat =
            dominantRatio >= 0.70 &&
            p75 <= 24.0 &&
            p90 <= 64.0;

        string strategy =
            flat
                ? "FLAT_FILL"
                : "TELEA";

        string reason =
            flat
                ? "dominant_local_background"
                : dominantRatio < 0.70
                    ? "background_not_dominant"
                    : p75 > 24.0
                        ? "background_variance"
                        : "background_outliers";

        return new V2BackgroundReconstructionAudit(
            target.TextRegionId,
            target.Kind.ToString(),
            bounds,
            strategy,
            maskPixels,
            samples.Count,
            background.Item0,
            background.Item1,
            background.Item2,
            dominantRatio,
            p75,
            p90,
            flat,
            reason);
    }

    public static void ApplyFlatFill(
        Mat destination,
        Mat glyphMask,
        V2BackgroundReconstructionAudit audit)
    {
        if (!audit.FlatAccepted ||
            Cv2.CountNonZero(
                glyphMask) < 2)
        {
            return;
        }

        destination.SetTo(
            new Scalar(
                audit.BackgroundB,
                audit.BackgroundG,
                audit.BackgroundR),
            glyphMask);
    }

    public static double MeasureFilledBackgroundP85(
        Mat image,
        Mat glyphMask,
        V2BackgroundReconstructionAudit audit)
    {
        if (!audit.FlatAccepted)
            return -1;

        var distances =
            new List<double>();

        var expected =
            new Vec3b(
                (byte)audit.BackgroundB,
                (byte)audit.BackgroundG,
                (byte)audit.BackgroundR);

        Rect bounds =
            ClampRect(
                audit.Bounds,
                image.Cols,
                image.Rows);

        for (int y =
                 bounds.Top;
             y <
             bounds.Bottom;
             y++)
        {
            for (int x =
                     bounds.Left;
                 x <
             bounds.Right;
                 x++)
            {
                if (glyphMask.At<byte>(
                        y,
                        x) == 0)
                {
                    continue;
                }

                distances.Add(
                    ColorDistance(
                        image.At<Vec3b>(
                            y,
                            x),
                        expected));
            }
        }

        if (distances.Count == 0)
            return 0;

        distances.Sort();

        return Percentile(
            distances,
            0.85);
    }

    static V2BackgroundReconstructionAudit EmptyAudit(
        V2TextTarget target,
        int maskPixels,
        string strategy,
        string reason,
        int sampleCount = 0)
        => new(
            target.TextRegionId,
            target.Kind.ToString(),
            target.TextBounds,
            strategy,
            maskPixels,
            sampleCount,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            reason);

    static double ColorDistance(
        Vec3b a,
        Vec3b b)
    {
        double db =
            a.Item0 -
            b.Item0;

        double dg =
            a.Item1 -
            b.Item1;

        double dr =
            a.Item2 -
            b.Item2;

        return Math.Sqrt(
            db * db +
            dg * dg +
            dr * dr);
    }

    static double Percentile(
        IReadOnlyList<double> sorted,
        double q)
    {
        if (sorted.Count == 0)
            return 0;

        int index =
            Math.Clamp(
                (int)Math.Round(
                    (sorted.Count - 1) *
                    q),
                0,
                sorted.Count - 1);

        return sorted[index];
    }

    static Rect ClampRect(
        Rect bounds,
        int cols,
        int rows)
    {
        int left =
            Math.Clamp(
                bounds.Left,
                0,
                cols);

        int top =
            Math.Clamp(
                bounds.Top,
                0,
                rows);

        int right =
            Math.Clamp(
                bounds.Right,
                left,
                cols);

        int bottom =
            Math.Clamp(
                bounds.Bottom,
                top,
                rows);

        return new Rect(
            left,
            top,
            right - left,
            bottom - top);
    }
}
