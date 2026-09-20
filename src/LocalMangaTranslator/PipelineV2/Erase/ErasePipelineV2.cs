using System.Text.Json;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2EraseTargetAudit(
    string TextRegionId,
    int InitialMaskPixels,
    int ResidualBeforeRetryPixels,
    int RetryMaskPixels,
    int ResidualAfterRetryPixels,
    bool Retried,
    string Status,
    int SelectedResidualPixels,
    string SelectedPass);

public sealed record ErasePipelineV2Result(
    int DetectedTargetCount,
    int TargetCount,
    int MaskPixels,
    int ResidualBeforeRetryPixels,
    int RetryMaskPixels,
    int ResidualAfterRetryPixels,
    int RetryTargetCount,
    string DetectionDebugPath,
    string MaskDebugPath,
    string FirstCleanedDebugPath,
    string ResidualDebugPath,
    string CleanedDebugPath,
    string DetectionJsonPath,
    IReadOnlyList<V2EraseTargetAudit> TargetAudits);

/// <summary>
/// Main Pipeline V2 erase stage.
///
/// Geometry comes only from the immutable RT-DETR snapshot. OCR/translation
/// decide which detector targets are needed, but never redefine their bounds.
/// The erase pass then performs one residual review and, when necessary, one
/// tightly-clamped +1 px retry around the original glyph mask.
/// </summary>
public sealed class ErasePipelineV2
{
    const double InpaintRadius = 3.0;
    const int LosslessWebpQuality = 101;

    public ErasePipelineV2Result Run(
        string sourcePath,
        string outputDirectory,
        V2DetectionSnapshot snapshot,
        IReadOnlySet<string>? selectedTextRegionIds = null,
        CancellationToken token = default)
    {
        string debugDir =
            OutputDirectoryLayout.Debug(outputDirectory);

        Directory.CreateDirectory(debugDir);

        string name =
            Path.GetFileNameWithoutExtension(sourcePath);

        string detectionPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_00_detection.webp");

        string maskPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_01_text_mask.webp");

        string firstCleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_02_cleaned_first.webp");

        string residualPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03_residual_check.webp");

        string cleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04_cleaned_final.webp");

        string jsonPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_erase_audit.json");

        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
        {
            throw new InvalidOperationException(
                "Pipeline V2 erase를 위해 원본 이미지를 열 수 없습니다.");
        }

        var targets =
            snapshot.TextTargets
                .Where(x =>
                    selectedTextRegionIds is null ||
                    selectedTextRegionIds.Contains(
                        x.TextRegionId))
                .ToList();

        using var initialMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        var initialPixelsByTarget =
            new Dictionary<string, int>(
                StringComparer.Ordinal);

        foreach (var target in targets)
        {
            token.ThrowIfCancellationRequested();

            using var targetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            int targetPixels =
                Cv2.CountNonZero(targetMask);

            initialPixelsByTarget[target.TextRegionId] =
                targetPixels;

            Cv2.BitwiseOr(
                initialMask,
                targetMask,
                initialMask);
        }

        SaveDetectionDebug(
            source,
            snapshot,
            selectedTextRegionIds,
            detectionPath);

        SaveMaskDebug(
            source,
            initialMask,
            maskPath);

        int initialPixels =
            Cv2.CountNonZero(initialMask);

        using var firstCleaned =
            InpaintOnlyMaskedPixels(
                source,
                initialMask);

        // TextFree captions on a genuinely flat panel (for example white text
        // on a solid black narration box) are a poor fit for Telea alone:
        // the interpolated edge can be re-segmented as "remaining text".
        // If the ring immediately around the detector box is low-variance,
        // replace only the already-approved glyph mask with that local panel
        // color. Detailed artwork never takes this path.
        ApplyFlatTextFreeBackgroundFill(
            source,
            firstCleaned,
            targets);

        SaveLosslessWebp(
            firstCleanedPath,
            firstCleaned);

        using var residualBefore =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var retryMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        var audits =
            new List<V2EraseTargetAudit>(
                targets.Count);

        int retryTargetCount = 0;

        foreach (var target in targets)
        {
            token.ThrowIfCancellationRequested();

            using var originalTargetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            using var residualCandidate =
                ComicTranslateComponentMask.Build(
                    firstCleaned,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: false);

            using var reviewZone =
                new Mat();

            using (var reviewKernel =
                   Cv2.GetStructuringElement(
                       MorphShapes.Ellipse,
                       new Size(5, 5)))
            {
                Cv2.Dilate(
                    originalTargetMask,
                    reviewZone,
                    reviewKernel,
                    iterations: 1);
            }

            using var residualNearOriginal =
                new Mat();

            Cv2.BitwiseAnd(
                residualCandidate,
                reviewZone,
                residualNearOriginal);

            int residualPixels =
                Cv2.CountNonZero(
                    residualNearOriginal);

            bool retry =
                !IsResidualAcceptable(
                    initialPixelsByTarget[
                        target.TextRegionId],
                    residualPixels,
                    target.TextBounds);

            int retryPixels = 0;

            if (retry)
            {
                retryTargetCount++;

                using var expanded =
                    new Mat();

                using (var oneMorePixelKernel =
                       Cv2.GetStructuringElement(
                           MorphShapes.Rect,
                           new Size(3, 3)))
                {
                    Cv2.Dilate(
                        originalTargetMask,
                        expanded,
                        oneMorePixelKernel,
                        iterations: 1);
                }

                using var targetBoundsMask =
                    BuildBoundsMask(
                        source.Rows,
                        source.Cols,
                        target.TextBounds);

                using var retryTargetMask =
                    new Mat();

                Cv2.BitwiseAnd(
                    expanded,
                    targetBoundsMask,
                    retryTargetMask);

                Cv2.BitwiseOr(
                    retryTargetMask,
                    residualNearOriginal,
                    retryTargetMask);

                Cv2.BitwiseAnd(
                    retryTargetMask,
                    targetBoundsMask,
                    retryTargetMask);

                retryPixels =
                    Cv2.CountNonZero(
                        retryTargetMask);

                Cv2.BitwiseOr(
                    retryMask,
                    retryTargetMask,
                    retryMask);

                Cv2.BitwiseOr(
                    residualBefore,
                    residualNearOriginal,
                    residualBefore);
            }

            int initialTargetPixels =
                initialPixelsByTarget.GetValueOrDefault(
                    target.TextRegionId);

            audits.Add(
                new V2EraseTargetAudit(
                    target.TextRegionId,
                    initialTargetPixels,
                    residualPixels,
                    retryPixels,
                    0,
                    retry,
                    initialTargetPixels < 2
                        ? "mask_empty"
                        : retry
                            ? "retry_scheduled"
                            : "clean_after_first_pass",
                    residualPixels,
                    retry
                        ? "pending_retry"
                        : "first"));
        }

        SaveResidualDebug(
            firstCleaned,
            residualBefore,
            residualPath);

        int retryMaskPixels =
            Cv2.CountNonZero(
                retryMask);

        using var finalCleaned =
            retryMaskPixels > 0
                ? InpaintOnlyMaskedPixels(
                    firstCleaned,
                    retryMask)
                : firstCleaned.Clone();

        int residualAfterTotal = 0;

        for (int i = 0; i < targets.Count; i++)
        {
            token.ThrowIfCancellationRequested();

            var target =
                targets[i];

            using var originalTargetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            using var finalResidualCandidate =
                ComicTranslateComponentMask.Build(
                    finalCleaned,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: false);

            using var finalReviewZone =
                new Mat();

            using (var finalKernel =
                   Cv2.GetStructuringElement(
                       MorphShapes.Ellipse,
                       new Size(5, 5)))
            {
                Cv2.Dilate(
                    originalTargetMask,
                    finalReviewZone,
                    finalKernel,
                    iterations: 1);
            }

            using var finalResidual =
                new Mat();

            Cv2.BitwiseAnd(
                finalResidualCandidate,
                finalReviewZone,
                finalResidual);

            int remainingAfterRetry =
                Cv2.CountNonZero(
                    finalResidual);

            var previous =
                audits[i];

            bool selectFirstPass =
                previous.Retried &&
                previous.ResidualBeforeRetryPixels <
                    remainingAfterRetry;

            int selectedResidual =
                selectFirstPass
                    ? previous.ResidualBeforeRetryPixels
                    : remainingAfterRetry;

            string selectedPass =
                selectFirstPass ||
                !previous.Retried
                    ? "first"
                    : "retry";

            if (selectFirstPass)
            {
                // A retry is only an optional improvement. If it makes the
                // residual score worse, restore exactly the retry footprint
                // from the first-pass image instead of committing the
                // regression.
                using var firstResidualCandidate =
                    ComicTranslateComponentMask.Build(
                        firstCleaned,
                        target.TextBounds,
                        target.BubbleBounds,
                        includeColorRescue: false);

                using var firstReviewZone =
                    new Mat();

                using (var firstReviewKernel =
                       Cv2.GetStructuringElement(
                           MorphShapes.Ellipse,
                           new Size(5, 5)))
                {
                    Cv2.Dilate(
                        originalTargetMask,
                        firstReviewZone,
                        firstReviewKernel,
                        iterations: 1);
                }

                using var firstResidual =
                    new Mat();

                Cv2.BitwiseAnd(
                    firstResidualCandidate,
                    firstReviewZone,
                    firstResidual);

                using var expanded =
                    new Mat();

                using (var oneMorePixelKernel =
                       Cv2.GetStructuringElement(
                           MorphShapes.Rect,
                           new Size(3, 3)))
                {
                    Cv2.Dilate(
                        originalTargetMask,
                        expanded,
                        oneMorePixelKernel,
                        iterations: 1);
                }

                using var targetBoundsMask =
                    BuildBoundsMask(
                        source.Rows,
                        source.Cols,
                        target.TextBounds);

                using var retryTargetMask =
                    new Mat();

                Cv2.BitwiseAnd(
                    expanded,
                    targetBoundsMask,
                    retryTargetMask);

                Cv2.BitwiseOr(
                    retryTargetMask,
                    firstResidual,
                    retryTargetMask);

                Cv2.BitwiseAnd(
                    retryTargetMask,
                    targetBoundsMask,
                    retryTargetMask);

                firstCleaned.CopyTo(
                    finalCleaned,
                    retryTargetMask);
            }

            residualAfterTotal +=
                selectedResidual;

            audits[i] =
                previous with
                {
                    ResidualAfterRetryPixels =
                        remainingAfterRetry,
                    SelectedResidualPixels =
                        selectedResidual,
                    SelectedPass =
                        selectedPass,
                    Status =
                        previous.InitialMaskPixels < 2
                            ? "mask_empty"
                            : IsResidualAcceptable(
                                previous.InitialMaskPixels,
                                selectedResidual,
                                target.TextBounds)
                                ? selectedPass == "retry"
                                    ? "clean_after_retry"
                                    : "clean_after_first_pass"
                                : "review_required"
                };
        }

        SaveLosslessWebp(
            cleanedPath,
            finalCleaned);

        var diagnostic =
            new
            {
                Schema =
                    "pipeline-v2-erase-audit-v2",
                SourceFile =
                    Path.GetFileName(sourcePath),
                snapshot.SourceMode,
                DetectedTargetCount =
                    snapshot.TextTargets.Count,
                SelectedTargetCount =
                    targets.Count,
                InitialMaskPixels =
                    initialPixels,
                ResidualBeforeRetryPixels =
                    audits.Sum(x =>
                        x.ResidualBeforeRetryPixels),
                RetryMaskPixels =
                    retryMaskPixels,
                ResidualAfterRetryPixels =
                    residualAfterTotal,
                RetryTargetCount =
                    retryTargetCount,
                Clean =
                    audits.All(IsAuditClean),
                Targets =
                    targets.Select(target =>
                    {
                        var audit =
                            audits.First(x =>
                                string.Equals(
                                    x.TextRegionId,
                                    target.TextRegionId,
                                    StringComparison.Ordinal));

                        return new
                        {
                            target.TextRegionId,
                            target.Kind,
                            TextBounds =
                                new
                                {
                                    target.TextBounds.X,
                                    target.TextBounds.Y,
                                    target.TextBounds.Width,
                                    target.TextBounds.Height
                                },
                            target.TextScore,
                            target.BubbleRegionId,
                            BubbleBounds =
                                target.BubbleBounds is Rect b
                                    ? new
                                    {
                                        b.X,
                                        b.Y,
                                        b.Width,
                                        b.Height
                                    }
                                    : null,
                            target.BubbleScore,
                            audit.InitialMaskPixels,
                            audit.ResidualBeforeRetryPixels,
                            audit.RetryMaskPixels,
                            audit.ResidualAfterRetryPixels,
                            audit.SelectedResidualPixels,
                            audit.SelectedPass,
                            audit.Retried,
                            audit.Status
                        };
                    })
            };

        File.WriteAllText(
            jsonPath,
            JsonSerializer.Serialize(
                diagnostic,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));

        return new ErasePipelineV2Result(
            snapshot.TextTargets.Count,
            targets.Count,
            initialPixels,
            audits.Sum(x =>
                x.ResidualBeforeRetryPixels),
            retryMaskPixels,
            residualAfterTotal,
            retryTargetCount,
            detectionPath,
            maskPath,
            firstCleanedPath,
            residualPath,
            cleanedPath,
            jsonPath,
            audits.ToArray());
    }

    public static bool IsAuditClean(
        V2EraseTargetAudit audit)
        => audit.InitialMaskPixels >= 2 &&
           audit.Status.StartsWith(
               "clean_",
               StringComparison.Ordinal);

    public static bool IsResidualAcceptable(
        int initialMaskPixels,
        int residualPixels,
        Rect textBounds)
    {
        if (residualPixels < 2)
            return true;

        int textArea =
            Math.Max(
                1,
                textBounds.Width *
                textBounds.Height);

        // 0019 showed that a few dozen high-contrast inpaint speckles can be
        // re-segmented as "text" even when the lettering is visibly gone.
        // Judge residuals by both original glyph-mask size and detector area
        // instead of the old absolute <2 pixel rule.
        double byOriginalMask =
            Math.Max(
                12.0,
                initialMaskPixels *
                0.10);

        double byDetectorArea =
            Math.Max(
                12.0,
                textArea *
                0.035);

        double allowance =
            Math.Min(
                byOriginalMask,
                byDetectorArea);

        return residualPixels <=
               allowance;
    }

    static void ApplyFlatTextFreeBackgroundFill(
        Mat source,
        Mat cleaned,
        IReadOnlyList<V2TextTarget> targets)
    {
        foreach (var target in targets)
        {
            if (target.Kind !=
                LocalMangaTranslator.Models.PageRegionKind.TextFree)
            {
                continue;
            }

            if (target.BubbleBounds.HasValue)
                continue;

            if (!TryEstimateFlatBackground(
                    source,
                    target.TextBounds,
                    out var background))
            {
                continue;
            }

            using var targetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            if (Cv2.CountNonZero(
                    targetMask) < 2)
            {
                continue;
            }

            cleaned.SetTo(
                new Scalar(
                    background.Item0,
                    background.Item1,
                    background.Item2),
                targetMask);
        }
    }

    static bool TryEstimateFlatBackground(
        Mat source,
        Rect textBounds,
        out Vec3b background)
    {
        background =
            new Vec3b(
                0,
                0,
                0);

        const int ring = 12;

        int left =
            Math.Max(
                0,
                textBounds.Left - ring);

        int top =
            Math.Max(
                0,
                textBounds.Top - ring);

        int right =
            Math.Min(
                source.Cols,
                textBounds.Right + ring);

        int bottom =
            Math.Min(
                source.Rows,
                textBounds.Bottom + ring);

        var samples =
            new List<Vec3b>();

        for (int y = top;
             y < bottom;
             y += 2)
        {
            for (int x = left;
                 x < right;
                 x += 2)
            {
                bool insideText =
                    x >= textBounds.Left &&
                    x < textBounds.Right &&
                    y >= textBounds.Top &&
                    y < textBounds.Bottom;

                if (insideText)
                    continue;

                samples.Add(
                    source.At<Vec3b>(
                        y,
                        x));
            }
        }

        if (samples.Count < 24)
            return false;

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

        background =
            new Vec3b(
                blues[
                    blues.Length / 2],
                greens[
                    greens.Length / 2],
                reds[
                    reds.Length / 2]);

        var backgroundValue =
            background;

        var distances =
            samples
                .Select(x =>
                {
                    double db =
                        x.Item0 -
                        backgroundValue.Item0;

                    double dg =
                        x.Item1 -
                        backgroundValue.Item1;

                    double dr =
                        x.Item2 -
                        backgroundValue.Item2;

                    return Math.Sqrt(
                        db * db +
                        dg * dg +
                        dr * dr);
                })
                .OrderBy(x =>
                    x)
                .ToArray();

        double p85 =
            distances[
                Math.Clamp(
                    (int)Math.Round(
                        (distances.Length - 1) *
                        0.85),
                    0,
                    distances.Length - 1)];

        // Flat enough for a panel-color fill. The threshold is intentionally
        // strict so free text over artwork, gradients or SFX remains on the
        // ordinary inpaint/review path.
        return p85 <= 34.0;
    }

    static Mat InpaintOnlyMaskedPixels(
        Mat source,
        Mat mask)
    {
        var cleaned =
            source.Clone();

        if (Cv2.CountNonZero(mask) == 0)
            return cleaned;

        using var inpainted =
            new Mat();

        Cv2.Inpaint(
            source,
            mask,
            inpainted,
            InpaintRadius,
            InpaintTypes.Telea);

        inpainted.CopyTo(
            cleaned,
            mask);

        return cleaned;
    }

    static Mat BuildBoundsMask(
        int rows,
        int cols,
        Rect bounds)
    {
        var mask =
            Mat.Zeros(
                rows,
                cols,
                MatType.CV_8UC1)
            .ToMat();

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

        if (right > left &&
            bottom > top)
        {
            Cv2.Rectangle(
                mask,
                new Rect(
                    left,
                    top,
                    right - left,
                    bottom - top),
                Scalar.White,
                thickness: -1);
        }

        return mask;
    }

    static void SaveDetectionDebug(
        Mat source,
        V2DetectionSnapshot snapshot,
        IReadOnlySet<string>? selectedTextRegionIds,
        string path)
    {
        using var debug =
            source.Clone();

        foreach (var region in snapshot.RawRegions)
        {
            bool selected =
                (region.Kind is
                    LocalMangaTranslator.Models.PageRegionKind.TextBubble or
                    LocalMangaTranslator.Models.PageRegionKind.TextFree) &&
                (selectedTextRegionIds is null ||
                 selectedTextRegionIds.Contains(
                     region.RegionId));

            Scalar color =
                region.Kind switch
                {
                    LocalMangaTranslator.Models.PageRegionKind.Bubble =>
                        new Scalar(0, 220, 220),

                    LocalMangaTranslator.Models.PageRegionKind.TextBubble
                        when selected =>
                        new Scalar(0, 0, 255),

                    LocalMangaTranslator.Models.PageRegionKind.TextFree
                        when selected =>
                        new Scalar(255, 80, 0),

                    LocalMangaTranslator.Models.PageRegionKind.TextBubble =>
                        new Scalar(110, 110, 110),

                    LocalMangaTranslator.Models.PageRegionKind.TextFree =>
                        new Scalar(160, 110, 60),

                    _ =>
                        new Scalar(130, 130, 130)
                };

            int thickness =
                selected
                    ? 3
                    : region.Kind ==
                      LocalMangaTranslator.Models.PageRegionKind.Bubble
                        ? 2
                        : 1;

            Cv2.Rectangle(
                debug,
                region.Bounds,
                color,
                thickness);
        }

        SaveLosslessWebp(
            path,
            debug);
    }

    static void SaveMaskDebug(
        Mat source,
        Mat mask,
        string path)
    {
        using var debug =
            source.Clone();

        using var red =
            new Mat(
                source.Rows,
                source.Cols,
                MatType.CV_8UC3,
                new Scalar(0, 0, 255));

        red.CopyTo(
            debug,
            mask);

        SaveLosslessWebp(
            path,
            debug);
    }

    static void SaveResidualDebug(
        Mat cleaned,
        Mat residualMask,
        string path)
    {
        using var debug =
            cleaned.Clone();

        using var red =
            new Mat(
                cleaned.Rows,
                cleaned.Cols,
                MatType.CV_8UC3,
                new Scalar(0, 0, 255));

        red.CopyTo(
            debug,
            residualMask);

        SaveLosslessWebp(
            path,
            debug);
    }

    static void SaveLosslessWebp(
        string path,
        Mat image)
    {
        if (!Cv2.ImWrite(
                path,
                image,
                [
                    new ImageEncodingParam(
                        ImwriteFlags.WebPQuality,
                        LosslessWebpQuality)
                ]))
        {
            throw new InvalidOperationException(
                $"Pipeline V2 디버그 이미지를 저장하지 못했습니다: {Path.GetFileName(path)}");
        }
    }
}
