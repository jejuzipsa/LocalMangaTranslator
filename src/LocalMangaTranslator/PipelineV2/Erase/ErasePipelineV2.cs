using System.Text.Json;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2EraseTargetAudit(
    string TextRegionId,
    int InitialMaskPixels,
    int CoreMaskPixels,
    int ResidualBeforeRetryPixels,
    int EffectiveResidualBeforeRetryPixels,
    int CoreOverlapBeforeRetryPixels,
    int PersistentCoreBeforeRetryPixels,
    int RetryMaskPixels,
    int ResidualAfterRetryPixels,
    int EffectiveResidualAfterRetryPixels,
    int CoreOverlapAfterRetryPixels,
    int PersistentCoreAfterRetryPixels,
    bool Retried,
    string Status,
    int SelectedResidualPixels,
    int SelectedPersistentCorePixels,
    double SelectedPersistenceRatio,
    string SelectedPass,
    string ReviewReason);

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
/// The erase pass then performs one source-glyph persistence review and, when
/// necessary, one tightly-clamped +1 px retry around the original erase mask.
/// Nearby bubble borders/artwork are diagnostic only unless they intersect the
/// immutable undilated source glyph core and still resemble the source pixels.
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

            using var originalCoreMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: true,
                    dilateMask: false);

            int coreMaskPixels =
                Cv2.CountNonZero(
                    originalCoreMask);

            using var residualCandidate =
                ComicTranslateComponentMask.Build(
                    firstCleaned,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: false,
                    dilateMask: false);

            using var reviewZone =
                new Mat();

            using (var reviewKernel =
                   Cv2.GetStructuringElement(
                       MorphShapes.Ellipse,
                       new Size(7, 7)))
            {
                Cv2.Dilate(
                    originalCoreMask,
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

            // 0024 kept this metric for diagnosis, but it is no longer a
            // commit criterion. Bubble borders and artwork live outside the
            // source glyph core and can otherwise look like false residue.
            using var effectiveResidual =
                BuildResidualOutsideOriginalMask(
                    residualNearOriginal,
                    originalCoreMask);

            int effectiveResidualPixels =
                Cv2.CountNonZero(
                    effectiveResidual);

            using var coreLinkedResidual =
                BuildCoreLinkedResidual(
                    residualNearOriginal,
                    originalCoreMask);

            int coreOverlapPixels =
                CountMaskOverlap(
                    coreLinkedResidual,
                    originalCoreMask);

            int persistentCorePixels =
                CountOriginalGlyphPersistence(
                    source,
                    firstCleaned,
                    originalCoreMask,
                    coreLinkedResidual,
                    target.TextBounds);

            bool retry =
                !IsCorePersistenceAcceptable(
                    coreMaskPixels,
                    coreOverlapPixels,
                    persistentCorePixels);

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
                    coreLinkedResidual,
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
                    coreLinkedResidual,
                    residualBefore);
            }

            int initialTargetPixels =
                initialPixelsByTarget.GetValueOrDefault(
                    target.TextRegionId);

            audits.Add(
                new V2EraseTargetAudit(
                    target.TextRegionId,
                    initialTargetPixels,
                    coreMaskPixels,
                    residualPixels,
                    effectiveResidualPixels,
                    coreOverlapPixels,
                    persistentCorePixels,
                    retryPixels,
                    0,
                    0,
                    0,
                    0,
                    retry,
                    initialTargetPixels < 2 ||
                    coreMaskPixels < 2
                        ? "mask_empty"
                        : retry
                            ? "retry_scheduled"
                            : "clean_after_first_pass",
                    residualPixels,
                    persistentCorePixels,
                    PersistenceRatio(
                        persistentCorePixels,
                        coreMaskPixels),
                    retry
                        ? "pending_retry"
                        : "first",
                    initialTargetPixels < 2 ||
                    coreMaskPixels < 2
                        ? "mask_empty"
                        : retry
                            ? "original_glyph_persistence"
                            : ReviewReason(
                                coreOverlapPixels,
                                persistentCorePixels)));
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

            using var originalCoreMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: true,
                    dilateMask: false);

            using var finalResidualCandidate =
                ComicTranslateComponentMask.Build(
                    finalCleaned,
                    target.TextBounds,
                    target.BubbleBounds,
                    includeColorRescue: false,
                    dilateMask: false);

            using var finalReviewZone =
                new Mat();

            using (var finalKernel =
                   Cv2.GetStructuringElement(
                       MorphShapes.Ellipse,
                       new Size(7, 7)))
            {
                Cv2.Dilate(
                    originalCoreMask,
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

            using var effectiveFinalResidual =
                BuildResidualOutsideOriginalMask(
                    finalResidual,
                    originalCoreMask);

            int effectiveRemainingAfterRetry =
                Cv2.CountNonZero(
                    effectiveFinalResidual);

            using var finalCoreLinkedResidual =
                BuildCoreLinkedResidual(
                    finalResidual,
                    originalCoreMask);

            int finalCoreOverlapPixels =
                CountMaskOverlap(
                    finalCoreLinkedResidual,
                    originalCoreMask);

            int finalPersistentCorePixels =
                CountOriginalGlyphPersistence(
                    source,
                    finalCleaned,
                    originalCoreMask,
                    finalCoreLinkedResidual,
                    target.TextBounds);

            var previous =
                audits[i];

            var selection =
                SelectBestResidualPass(
                    previous.PersistentCoreBeforeRetryPixels,
                    finalPersistentCorePixels,
                    previous.Retried);

            bool selectFirstPass =
                selection.SelectedPass ==
                "first" &&
                previous.Retried;

            int selectedResidual =
                selection.SelectedResidualPixels;

            string selectedPass =
                selection.SelectedPass;

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
                        includeColorRescue: false,
                        dilateMask: false);

                using var firstReviewZone =
                    new Mat();

                using (var firstReviewKernel =
                       Cv2.GetStructuringElement(
                           MorphShapes.Ellipse,
                           new Size(7, 7)))
                {
                    Cv2.Dilate(
                        originalCoreMask,
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

                using var firstCoreLinkedResidual =
                    BuildCoreLinkedResidual(
                        firstResidual,
                        originalCoreMask);

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
                    firstCoreLinkedResidual,
                    retryTargetMask);

                Cv2.BitwiseAnd(
                    retryTargetMask,
                    targetBoundsMask,
                    retryTargetMask);

                firstCleaned.CopyTo(
                    finalCleaned,
                    retryTargetMask);
            }

            int selectedRawResidual =
                selectedPass == "retry"
                    ? remainingAfterRetry
                    : previous.ResidualBeforeRetryPixels;

            int selectedCoreOverlap =
                selectedPass == "retry"
                    ? finalCoreOverlapPixels
                    : previous.CoreOverlapBeforeRetryPixels;

            int selectedPersistentCore =
                selectedPass == "retry"
                    ? finalPersistentCorePixels
                    : previous.PersistentCoreBeforeRetryPixels;

            bool selectedClean =
                IsCorePersistenceAcceptable(
                    previous.CoreMaskPixels,
                    selectedCoreOverlap,
                    selectedPersistentCore);

            residualAfterTotal +=
                selectedRawResidual;

            audits[i] =
                previous with
                {
                    ResidualAfterRetryPixels =
                        remainingAfterRetry,
                    EffectiveResidualAfterRetryPixels =
                        effectiveRemainingAfterRetry,
                    CoreOverlapAfterRetryPixels =
                        finalCoreOverlapPixels,
                    PersistentCoreAfterRetryPixels =
                        finalPersistentCorePixels,
                    SelectedResidualPixels =
                        selectedRawResidual,
                    SelectedPersistentCorePixels =
                        selectedPersistentCore,
                    SelectedPersistenceRatio =
                        PersistenceRatio(
                            selectedPersistentCore,
                            previous.CoreMaskPixels),
                    SelectedPass =
                        selectedPass,
                    ReviewReason =
                        previous.InitialMaskPixels < 2 ||
                        previous.CoreMaskPixels < 2
                            ? "mask_empty"
                            : selectedClean
                                ? ReviewReason(
                                    selectedCoreOverlap,
                                    selectedPersistentCore)
                                : "original_glyph_persistence",
                    Status =
                        previous.InitialMaskPixels < 2 ||
                        previous.CoreMaskPixels < 2
                            ? "mask_empty"
                            : selectedClean
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
                            audit.CoreMaskPixels,
                            audit.ResidualBeforeRetryPixels,
                            audit.EffectiveResidualBeforeRetryPixels,
                            audit.CoreOverlapBeforeRetryPixels,
                            audit.PersistentCoreBeforeRetryPixels,
                            audit.RetryMaskPixels,
                            audit.ResidualAfterRetryPixels,
                            audit.EffectiveResidualAfterRetryPixels,
                            audit.CoreOverlapAfterRetryPixels,
                            audit.PersistentCoreAfterRetryPixels,
                            audit.SelectedResidualPixels,
                            audit.SelectedPersistentCorePixels,
                            audit.SelectedPersistenceRatio,
                            audit.SelectedPass,
                            audit.Retried,
                            audit.ReviewReason,
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

    public static (int SelectedResidualPixels, string SelectedPass)
        SelectBestResidualPass(
            int firstResidualPixels,
            int retryResidualPixels,
            bool retried)
    {
        if (!retried ||
            firstResidualPixels <=
                retryResidualPixels)
        {
            return (
                firstResidualPixels,
                "first");
        }

        return (
            retryResidualPixels,
            "retry");
    }

    public static bool IsAuditClean(
        V2EraseTargetAudit audit)
        => audit.InitialMaskPixels >= 2 &&
           audit.Status.StartsWith(
               "clean_",
               StringComparison.Ordinal);

    public static bool IsCorePersistenceAcceptable(
        int coreMaskPixels,
        int coreOverlapPixels,
        int persistentCorePixels)
    {
        if (coreMaskPixels < 2)
            return false;

        if (coreOverlapPixels < 2 ||
            persistentCorePixels < 2)
        {
            return true;
        }

        int persistenceAllowance =
            Math.Max(
                2,
                Math.Min(
                    12,
                    (int)Math.Ceiling(
                        coreMaskPixels *
                        0.005)));

        return persistentCorePixels <=
               persistenceAllowance;
    }

    public static int CountMaskOverlap(
        Mat a,
        Mat b)
    {
        using var overlap =
            new Mat();

        Cv2.BitwiseAnd(
            a,
            b,
            overlap);

        return Cv2.CountNonZero(
            overlap);
    }

    public static int CountOriginalGlyphPersistence(
        Mat source,
        Mat cleaned,
        Mat originalCoreMask,
        Mat coreLinkedResidual,
        Rect textBounds)
    {
        int left =
            Math.Clamp(
                textBounds.Left,
                0,
                source.Cols);

        int top =
            Math.Clamp(
                textBounds.Top,
                0,
                source.Rows);

        int right =
            Math.Clamp(
                textBounds.Right,
                left,
                source.Cols);

        int bottom =
            Math.Clamp(
                textBounds.Bottom,
                top,
                source.Rows);

        int persistent =
            0;

        const double sameGlyphDistance =
            28.0;

        for (int y = top;
             y < bottom;
             y++)
        {
            for (int x = left;
                 x < right;
                 x++)
            {
                if (originalCoreMask.At<byte>(
                        y,
                        x) == 0 ||
                    coreLinkedResidual.At<byte>(
                        y,
                        x) == 0)
                {
                    continue;
                }

                var before =
                    source.At<Vec3b>(
                        y,
                        x);

                var after =
                    cleaned.At<Vec3b>(
                        y,
                        x);

                double db =
                    before.Item0 -
                    after.Item0;

                double dg =
                    before.Item1 -
                    after.Item1;

                double dr =
                    before.Item2 -
                    after.Item2;

                double distance =
                    Math.Sqrt(
                        db * db +
                        dg * dg +
                        dr * dr);

                if (distance <=
                    sameGlyphDistance)
                {
                    persistent++;
                }
            }
        }

        return persistent;
    }

    public static Mat BuildCoreLinkedResidual(
        Mat residualNearOriginal,
        Mat originalCoreMask)
    {
        var linked =
            Mat.Zeros(
                residualNearOriginal.Rows,
                residualNearOriginal.Cols,
                MatType.CV_8UC1)
            .ToMat();

        Cv2.FindContours(
            residualNearOriginal,
            out Point[][] contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        int corePixels =
            Math.Max(
                1,
                Cv2.CountNonZero(
                    originalCoreMask));

        foreach (var contour in contours)
        {
            if (contour.Length == 0)
                continue;

            using var component =
                Mat.Zeros(
                    residualNearOriginal.Rows,
                    residualNearOriginal.Cols,
                    MatType.CV_8UC1)
                .ToMat();

            Cv2.DrawContours(
                component,
                [contour],
                -1,
                Scalar.White,
                thickness: -1);

            int componentPixels =
                Cv2.CountNonZero(
                    component);

            if (componentPixels < 2)
                continue;

            int overlapPixels =
                CountMaskOverlap(
                    component,
                    originalCoreMask);

            if (overlapPixels < 2)
                continue;

            double componentOverlapRatio =
                overlapPixels /
                (double)Math.Max(
                    1,
                    componentPixels);

            double coreOverlapRatio =
                overlapPixels /
                (double)corePixels;

            // A bubble border or artwork line may enter the review halo.
            // Keep it only when it genuinely intersects the source glyph
            // core, rather than merely passing close to the text.
            if (componentOverlapRatio < 0.08 &&
                coreOverlapRatio < 0.01 &&
                overlapPixels < 12)
            {
                continue;
            }

            Cv2.BitwiseOr(
                linked,
                component,
                linked);
        }

        return linked;
    }

    static double PersistenceRatio(
        int persistentCorePixels,
        int coreMaskPixels)
        => coreMaskPixels <= 0
            ? 0
            : persistentCorePixels /
              (double)coreMaskPixels;

    static string ReviewReason(
        int coreOverlapPixels,
        int persistentCorePixels)
    {
        if (coreOverlapPixels < 2)
            return "surrounding_structure_only";

        if (persistentCorePixels < 2)
            return "core_changed_after_erase";

        return "core_persistence_below_allowance";
    }

    public static bool IsResidualAcceptable(
        int initialMaskPixels,
        int residualPixels,
        Rect textBounds)
        => IsResidualAcceptable(
            initialMaskPixels,
            residualPixels,
            residualPixels,
            textBounds);

    public static bool IsResidualAcceptable(
        int initialMaskPixels,
        int rawResidualPixels,
        int effectiveResidualPixels,
        Rect textBounds)
    {
        if (rawResidualPixels < 2)
            return true;

        int textArea =
            Math.Max(
                1,
                textBounds.Width *
                textBounds.Height);

        // The raw residual detector also re-segments contrast created *inside*
        // pixels that were already erased/inpainted. Those pixels cannot be
        // surviving source glyphs, so use them only as a sanity bound.
        // The decisive signal is text-like residue outside the immutable
        // original glyph mask but still inside its local review halo.
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

        double rawAllowance =
            Math.Min(
                byOriginalMask,
                byDetectorArea);

        if (rawResidualPixels <=
            rawAllowance)
        {
            return true;
        }

        double effectiveAllowance =
            Math.Max(
                6.0,
                rawAllowance *
                0.25);

        // A visually clean inpaint may still have substantial texture inside
        // the replaced glyph pixels. Accept that only when the suspicious
        // residue outside the erase mask is tiny and the raw detector is not
        // wildly inconsistent. This keeps true missed strokes reviewable.
        return rawResidualPixels <=
                   rawAllowance * 2.0 &&
               effectiveResidualPixels <=
                   effectiveAllowance;
    }

    public static int CountResidualOutsideOriginalMask(
        Mat residualNearOriginal,
        Mat originalTargetMask)
    {
        using var effective =
            BuildResidualOutsideOriginalMask(
                residualNearOriginal,
                originalTargetMask);

        return Cv2.CountNonZero(
            effective);
    }

    static Mat BuildResidualOutsideOriginalMask(
        Mat residualNearOriginal,
        Mat originalTargetMask)
    {
        var inverseOriginal =
            new Mat();

        Cv2.BitwiseNot(
            originalTargetMask,
            inverseOriginal);

        var effective =
            new Mat();

        Cv2.BitwiseAnd(
            residualNearOriginal,
            inverseOriginal,
            effective);

        inverseOriginal.Dispose();

        return effective;
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
