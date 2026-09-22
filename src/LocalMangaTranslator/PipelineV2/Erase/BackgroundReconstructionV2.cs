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
    double SpatialCoverage,
    int ExclusionRadius,
    bool FlatAccepted,
    string Reason);

/// <summary>
/// Chooses how erased comic lettering is reconstructed.
///
/// A text mask describes foreground ink. It does not describe what should
/// replace that ink. For locally flat Bubble/Caption backgrounds we restore
/// the dominant local background cluster directly. Complex artwork remains on
/// the ordinary inpaint path.
///
/// 0031 intentionally purifies background candidates before classifying them:
/// the accepted glyph mask is expanded to exclude antialias/outline/color
/// contamination, then the largest compact color cluster is selected instead
/// of taking the median of every remaining pixel.
/// </summary>
public static class BackgroundReconstructionV2
{
    const double SeedDistance = 54.0;
    const double MatchDistance = 34.0;
    const int QuantizationStep = 32;
    const int CandidateBinCount = 8;

    readonly record struct SamplePoint(
        int X,
        int Y,
        Vec3b Color);

    sealed record ClusterCandidate(
        Vec3b Background,
        int SupportCount,
        double DominantRatio,
        double P75,
        double P90,
        double SpatialCoverage);

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

        int exclusionRadius =
            Math.Clamp(
                (int)Math.Round(
                    Math.Min(
                        bounds.Width,
                        bounds.Height) *
                    0.035),
                3,
                9);

        var samples =
            CollectPurifiedSamples(
                source,
                glyphMask,
                sampleBounds,
                exclusionRadius);

        int minimumSamples =
            Math.Max(
                32,
                (sampleBounds.Width *
                 sampleBounds.Height) /
                180);

        if (samples.Count <
            minimumSamples &&
            exclusionRadius > 2)
        {
            exclusionRadius =
                Math.Max(
                    2,
                    exclusionRadius / 2);

            samples =
                CollectPurifiedSamples(
                    source,
                    glyphMask,
                    sampleBounds,
                    exclusionRadius);
        }

        if (samples.Count <
            minimumSamples)
        {
            return EmptyAudit(
                target,
                maskPixels,
                "TELEA",
                "insufficient_purified_background_samples",
                samples.Count,
                exclusionRadius);
        }

        var cluster =
            FindDominantCluster(
                samples,
                sampleBounds);

        if (cluster is null)
        {
            return EmptyAudit(
                target,
                maskPixels,
                "TELEA",
                "no_background_cluster",
                samples.Count,
                exclusionRadius);
        }

        bool flat =
            cluster.DominantRatio >= 0.60 &&
            cluster.P90 <= 30.0 &&
            cluster.SpatialCoverage >= 0.50;

        string strategy =
            flat
                ? "FLAT_FILL"
                : "TELEA";

        string reason =
            flat
                ? "dominant_purified_background_cluster"
                : cluster.DominantRatio < 0.60
                    ? "background_cluster_weak"
                    : cluster.P90 > 30.0
                        ? "background_cluster_variance"
                        : "background_cluster_not_spatial";

        return new V2BackgroundReconstructionAudit(
            target.TextRegionId,
            target.Kind.ToString(),
            bounds,
            strategy,
            maskPixels,
            samples.Count,
            cluster.Background.Item0,
            cluster.Background.Item1,
            cluster.Background.Item2,
            cluster.DominantRatio,
            cluster.P75,
            cluster.P90,
            cluster.SpatialCoverage,
            exclusionRadius,
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

    static List<SamplePoint> CollectPurifiedSamples(
        Mat source,
        Mat glyphMask,
        Rect sampleBounds,
        int exclusionRadius)
    {
        using var exclusion =
            new Mat();

        using (var kernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Ellipse,
                   new Size(
                       exclusionRadius * 2 + 1,
                       exclusionRadius * 2 + 1)))
        {
            Cv2.Dilate(
                glyphMask,
                exclusion,
                kernel,
                iterations: 1);
        }

        var samples =
            new List<SamplePoint>();

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
                if (exclusion.At<byte>(
                        y,
                        x) != 0)
                {
                    continue;
                }

                samples.Add(
                    new SamplePoint(
                        x,
                        y,
                        source.At<Vec3b>(
                            y,
                            x)));
            }
        }

        return samples;
    }

    static ClusterCandidate? FindDominantCluster(
        IReadOnlyList<SamplePoint> samples,
        Rect sampleBounds)
    {
        var bins =
            new Dictionary<int, List<SamplePoint>>();

        foreach (var sample in samples)
        {
            int key =
                QuantizedKey(
                    sample.Color);

            if (!bins.TryGetValue(
                    key,
                    out var bucket))
            {
                bucket =
                    new List<SamplePoint>();

                bins[key] =
                    bucket;
            }

            bucket.Add(
                sample);
        }

        var seedBins =
            bins.Values
                .OrderByDescending(x =>
                    x.Count)
                .Take(
                    CandidateBinCount)
                .ToArray();

        ClusterCandidate? best =
            null;

        foreach (var seedBin in seedBins)
        {
            var seed =
                ChannelMedian(
                    seedBin.Select(x =>
                        x.Color));

            var broadSupport =
                samples
                    .Where(x =>
                        ColorDistance(
                            x.Color,
                            seed) <=
                        SeedDistance)
                    .ToArray();

            if (broadSupport.Length < 8)
                continue;

            var refined =
                ChannelMedian(
                    broadSupport.Select(x =>
                        x.Color));

            var support =
                samples
                    .Where(x =>
                        ColorDistance(
                            x.Color,
                            refined) <=
                        MatchDistance)
                    .ToArray();

            if (support.Length < 8)
                continue;

            refined =
                ChannelMedian(
                    support.Select(x =>
                        x.Color));

            var distances =
                support
                    .Select(x =>
                        ColorDistance(
                            x.Color,
                            refined))
                    .OrderBy(x =>
                        x)
                    .ToArray();

            double ratio =
                support.Length /
                (double)Math.Max(
                    1,
                    samples.Count);

            double spatial =
                ComputeSpatialCoverage(
                    support,
                    sampleBounds);

            var candidate =
                new ClusterCandidate(
                    refined,
                    support.Length,
                    ratio,
                    Percentile(
                        distances,
                        0.75),
                    Percentile(
                        distances,
                        0.90),
                    spatial);

            if (best is null ||
                candidate.SupportCount >
                best.SupportCount ||
                candidate.SupportCount ==
                best.SupportCount &&
                candidate.SpatialCoverage >
                best.SpatialCoverage ||
                candidate.SupportCount ==
                best.SupportCount &&
                Math.Abs(
                    candidate.SpatialCoverage -
                    best.SpatialCoverage) <
                0.0001 &&
                candidate.P90 <
                best.P90)
            {
                best =
                    candidate;
            }
        }

        return best;
    }

    static double ComputeSpatialCoverage(
        IReadOnlyList<SamplePoint> support,
        Rect bounds)
    {
        const int grid =
            4;

        var occupied =
            new bool[
                grid,
                grid];

        foreach (var sample in support)
        {
            int gx =
                Math.Clamp(
                    (int)(
                        (sample.X -
                         bounds.Left) /
                        (double)Math.Max(
                            1,
                            bounds.Width) *
                        grid),
                    0,
                    grid - 1);

            int gy =
                Math.Clamp(
                    (int)(
                        (sample.Y -
                         bounds.Top) /
                        (double)Math.Max(
                            1,
                            bounds.Height) *
                        grid),
                    0,
                    grid - 1);

            occupied[
                gx,
                gy] =
                true;
        }

        int count =
            0;

        for (int y = 0;
             y < grid;
             y++)
        {
            for (int x = 0;
                 x < grid;
                 x++)
            {
                if (occupied[
                        x,
                        y])
                {
                    count++;
                }
            }
        }

        return count /
               (double)(
                   grid *
                   grid);
    }

    static Vec3b ChannelMedian(
        IEnumerable<Vec3b> values)
    {
        var colors =
            values.ToArray();

        if (colors.Length == 0)
        {
            return new Vec3b(
                0,
                0,
                0);
        }

        var blues =
            colors
                .Select(x =>
                    x.Item0)
                .OrderBy(x =>
                    x)
                .ToArray();

        var greens =
            colors
                .Select(x =>
                    x.Item1)
                .OrderBy(x =>
                    x)
                .ToArray();

        var reds =
            colors
                .Select(x =>
                    x.Item2)
                .OrderBy(x =>
                    x)
                .ToArray();

        int middle =
            colors.Length /
            2;

        return new Vec3b(
            blues[middle],
            greens[middle],
            reds[middle]);
    }

    static int QuantizedKey(
        Vec3b color)
    {
        int b =
            color.Item0 /
            QuantizationStep;

        int g =
            color.Item1 /
            QuantizationStep;

        int r =
            color.Item2 /
            QuantizationStep;

        return
            b |
            g << 4 |
            r << 8;
    }

    static V2BackgroundReconstructionAudit EmptyAudit(
        V2TextTarget target,
        int maskPixels,
        string strategy,
        string reason,
        int sampleCount = 0,
        int exclusionRadius = 0)
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
            0,
            exclusionRadius,
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
