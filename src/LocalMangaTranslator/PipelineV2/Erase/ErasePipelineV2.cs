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
    string PrimaryReview,
    string SecondaryReview,
    bool RescuedBySecondary,
    string ReviewReason);

public sealed record V2CleanedCheckpointAudit(
    string CheckpointId,
    string Kind,
    Rect Bounds,
    IReadOnlyList<string> TextRegionIds,
    bool EmptyVerified,
    string Status);

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
    string LegacyReviewerCheckpointDebugPath,
    string DetectionJsonPath,
    IReadOnlyList<V2EraseTargetAudit> TargetAudits,
    IReadOnlyList<V2CleanedCheckpointAudit> CleanedCheckpointAudits,
    IReadOnlyList<string> BackgroundQualityFailedTextRegionIds);

/// <summary>
/// Main Pipeline V2 erase stage.
///
/// Geometry comes only from the immutable RT-DETR snapshot. OCR/translation
/// decide which detector targets are needed, but never redefine their bounds.
/// 0026 keeps the proven 0024 density review as the primary gate. Only a
/// target that still fails that gate after best-pass selection is allowed into
/// the 0025 source-glyph persistence review as a rescue check. A primary pass
/// can never be overturned by the secondary reviewer.
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

        string grayscaleMaskPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_01a_grayscale_mask.webp");

        string colorRescueMaskPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_01c_color_rescue_mask.webp");

        string firstCleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_02_cleaned_first.webp");

        string reconstructionDebugPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_02a_reconstruction_strategy.webp");

        string backgroundAuditPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_background_audit.json");

        string residualPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03_residual_check.webp");

        string glyphCorePath =
            Path.Combine(
                debugDir,
                $"{name}.v2_01b_glyph_core.webp");

        string coreLinkedResidualPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03a_core_linked_residual.webp");

        string persistentCorePath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03b_persistent_core.webp");

        string flatResidualOutlierPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03c_flat_residual_outlier.webp");

        string flatCleanupMaskPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_03d_flat_cleanup_mask.webp");

        string cleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04_cleaned_final.webp");

        string legacyCheckpointPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04a_legacy_reviewer_checkpoint.webp");

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

        using var grayscaleMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var colorRescueMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        var grayscalePixelsByTarget =
            new Dictionary<string, int>(
                StringComparer.Ordinal);

        var colorPixelsByTarget =
            new Dictionary<string, int>(
                StringComparer.Ordinal);

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

            using var targetGrayscaleMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds,
                    layer:
                        ComicTranslateMaskLayer.Grayscale);

            using var targetColorMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds,
                    layer:
                        ComicTranslateMaskLayer.ColorRescue);

            int targetPixels =
                Cv2.CountNonZero(
                    targetMask);

            int grayscalePixels =
                Cv2.CountNonZero(
                    targetGrayscaleMask);

            int colorPixels =
                Cv2.CountNonZero(
                    targetColorMask);

            initialPixelsByTarget[target.TextRegionId] =
                targetPixels;

            grayscalePixelsByTarget[target.TextRegionId] =
                grayscalePixels;

            colorPixelsByTarget[target.TextRegionId] =
                colorPixels;

            Cv2.BitwiseOr(
                initialMask,
                targetMask,
                initialMask);

            Cv2.BitwiseOr(
                grayscaleMask,
                targetGrayscaleMask,
                grayscaleMask);

            Cv2.BitwiseOr(
                colorRescueMask,
                targetColorMask,
                colorRescueMask);
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

        SaveColoredMaskDebug(
            source,
            grayscaleMask,
            new Scalar(
                0,
                255,
                255),
            grayscaleMaskPath);

        SaveColoredMaskDebug(
            source,
            colorRescueMask,
            new Scalar(
                255,
                0,
                255),
            colorRescueMaskPath);

        int initialPixels =
            Cv2.CountNonZero(initialMask);

        using var firstCleaned =
            ReconstructFirstPass(
                source,
                targets,
                out var backgroundAudits,
                out var backgroundQualityP85);

        SaveLosslessWebp(
            firstCleanedPath,
            firstCleaned);

        SaveBackgroundStrategyDebug(
            firstCleaned,
            backgroundAudits,
            backgroundQualityP85,
            reconstructionDebugPath);

        WriteBackgroundAudit(
            sourcePath,
            backgroundAudits,
            backgroundQualityP85,
            backgroundAuditPath);

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

        using var flatRetryMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var teleaRetryMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        var backgroundAuditByTarget =
            backgroundAudits.ToDictionary(
                x => x.TextRegionId,
                StringComparer.Ordinal);

        var retryMasksByTarget =
            new Dictionary<string, Mat>(
                StringComparer.Ordinal);

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
                       new Size(7, 7)))
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

            using var effectiveResidual =
                BuildResidualOutsideOriginalMask(
                    residualNearOriginal,
                    originalTargetMask);

            int effectiveResidualPixels =
                Cv2.CountNonZero(
                    effectiveResidual);

            int initialTargetPixels =
                initialPixelsByTarget.GetValueOrDefault(
                    target.TextRegionId);

            bool primaryClean =
                IsResidualAcceptable(
                    initialTargetPixels,
                    residualPixels,
                    effectiveResidualPixels,
                    target.TextBounds);

            bool retry =
                initialTargetPixels >= 2 &&
                !primaryClean;

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
                    effectiveResidual,
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

                retryMasksByTarget[
                    target.TextRegionId] =
                    retryTargetMask.Clone();

                Cv2.BitwiseOr(
                    residualBefore,
                    effectiveResidual,
                    residualBefore);
            }

            audits.Add(
                new V2EraseTargetAudit(
                    target.TextRegionId,
                    initialTargetPixels,
                    0,
                    residualPixels,
                    effectiveResidualPixels,
                    0,
                    0,
                    retryPixels,
                    0,
                    0,
                    0,
                    0,
                    retry,
                    initialTargetPixels < 2
                        ? "mask_empty"
                        : retry
                            ? "retry_scheduled"
                            : "clean_after_first_pass",
                    effectiveResidualPixels,
                    0,
                    0,
                    retry
                        ? "pending_retry"
                        : "first",
                    initialTargetPixels < 2
                        ? "mask_empty"
                        : primaryClean
                            ? "pass"
                            : "fail",
                    "not_run",
                    false,
                    initialTargetPixels < 2
                        ? "mask_empty"
                        : primaryClean
                            ? "primary_residual_clean"
                            : "primary_residual_retry"));
        }

        SaveResidualDebug(
            firstCleaned,
            residualBefore,
            residualPath);

        int retryMaskPixels =
            Cv2.CountNonZero(
                retryMask);

        int flatRetryTargetCount = 0;
        int teleaRetryTargetCount = 0;

        using var finalCleaned =
            firstCleaned.Clone();

        foreach (var target in targets)
        {
            if (!retryMasksByTarget.TryGetValue(
                    target.TextRegionId,
                    out var targetRetryMask))
            {
                continue;
            }

            if (!backgroundAuditByTarget.TryGetValue(
                    target.TextRegionId,
                    out var backgroundAudit))
            {
                Cv2.BitwiseOr(
                    teleaRetryMask,
                    targetRetryMask,
                    teleaRetryMask);

                teleaRetryTargetCount++;

                continue;
            }

            if (backgroundAudit.FlatAccepted)
            {
                // 0032: retry must preserve the reconstruction strategy chosen
                // for this immutable TextRegionId. In particular, a flat
                // balloon must never fall back to Telea, because surviving
                // colored glyph pixels would become inpaint source material.
                BackgroundReconstructionV2.ApplyFlatFill(
                    finalCleaned,
                    targetRetryMask,
                    backgroundAudit);

                Cv2.BitwiseOr(
                    flatRetryMask,
                    targetRetryMask,
                    flatRetryMask);

                flatRetryTargetCount++;
            }
            else
            {
                Cv2.BitwiseOr(
                    teleaRetryMask,
                    targetRetryMask,
                    teleaRetryMask);

                teleaRetryTargetCount++;
            }
        }

        // If detector boxes ever overlap, FLAT_FILL owns its pixels. Do not
        // let a neighboring TELEA retry sample or overwrite an already
        // reconstructed flat background.
        if (Cv2.CountNonZero(
                flatRetryMask) > 0 &&
            Cv2.CountNonZero(
                teleaRetryMask) > 0)
        {
            using var notFlatRetryMask =
                new Mat();

            Cv2.BitwiseNot(
                flatRetryMask,
                notFlatRetryMask);

            Cv2.BitwiseAnd(
                teleaRetryMask,
                notFlatRetryMask,
                teleaRetryMask);
        }

        int teleaRetryMaskPixels =
            Cv2.CountNonZero(
                teleaRetryMask);

        if (teleaRetryMaskPixels > 0)
        {
            using var teleaRetryCleaned =
                InpaintOnlyMaskedPixels(
                    finalCleaned,
                    teleaRetryMask);

            teleaRetryCleaned.CopyTo(
                finalCleaned);
        }

        int flatRetryMaskPixels =
            Cv2.CountNonZero(
                flatRetryMask);

        foreach (var targetRetryMask in
                 retryMasksByTarget.Values)
        {
            targetRetryMask.Dispose();
        }

        retryMasksByTarget.Clear();

        int residualAfterTotal = 0;

        using var secondaryCoreDebug =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var secondaryLinkedDebug =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var secondaryPersistentDebug =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

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
                       new Size(7, 7)))
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

            using var effectiveFinalResidual =
                BuildResidualOutsideOriginalMask(
                    finalResidual,
                    originalTargetMask);

            int effectiveRemainingAfterRetry =
                Cv2.CountNonZero(
                    effectiveFinalResidual);

            var previous =
                audits[i];

            var selection =
                SelectBestResidualPass(
                    previous.EffectiveResidualBeforeRetryPixels,
                    effectiveRemainingAfterRetry,
                    previous.Retried);

            bool selectFirstPass =
                selection.SelectedPass ==
                "first" &&
                previous.Retried;

            int selectedEffectiveResidual =
                selection.SelectedResidualPixels;

            string selectedPass =
                selection.SelectedPass;

            if (selectFirstPass)
            {
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
                           new Size(7, 7)))
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

                using var effectiveFirstResidual =
                    BuildResidualOutsideOriginalMask(
                        firstResidual,
                        originalTargetMask);

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
                    effectiveFirstResidual,
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

            bool primaryClean =
                previous.InitialMaskPixels >= 2 &&
                IsResidualAcceptable(
                    previous.InitialMaskPixels,
                    selectedRawResidual,
                    selectedEffectiveResidual,
                    target.TextBounds);

            int coreMaskPixels = 0;
            int coreOverlapPixels = 0;
            int persistentCorePixels = 0;
            bool secondaryClean = false;
            string secondaryReview =
                "not_run";

            if (!primaryClean &&
                previous.InitialMaskPixels >= 2)
            {
                using var originalCoreMask =
                    ComicTranslateComponentMask.Build(
                        source,
                        target.TextBounds,
                        target.BubbleBounds,
                        includeColorRescue: true,
                        dilateMask: false);

                coreMaskPixels =
                    Cv2.CountNonZero(
                        originalCoreMask);

                using var secondaryResidualCandidate =
                    ComicTranslateComponentMask.Build(
                        finalCleaned,
                        target.TextBounds,
                        target.BubbleBounds,
                        includeColorRescue: false,
                        dilateMask: false);

                using var secondaryReviewZone =
                    new Mat();

                using (var secondaryKernel =
                       Cv2.GetStructuringElement(
                           MorphShapes.Ellipse,
                           new Size(7, 7)))
                {
                    Cv2.Dilate(
                        originalCoreMask,
                        secondaryReviewZone,
                        secondaryKernel,
                        iterations: 1);
                }

                using var secondaryResidual =
                    new Mat();

                Cv2.BitwiseAnd(
                    secondaryResidualCandidate,
                    secondaryReviewZone,
                    secondaryResidual);

                using var coreLinkedResidual =
                    BuildCoreLinkedResidual(
                        secondaryResidual,
                        originalCoreMask);

                coreOverlapPixels =
                    CountMaskOverlap(
                        coreLinkedResidual,
                        originalCoreMask);

                using var persistentCoreMask =
                    BuildOriginalGlyphPersistenceMask(
                        source,
                        finalCleaned,
                        originalCoreMask,
                        coreLinkedResidual,
                        target.TextBounds);

                persistentCorePixels =
                    Cv2.CountNonZero(
                        persistentCoreMask);

                secondaryClean =
                    IsCorePersistenceAcceptable(
                        coreMaskPixels,
                        coreOverlapPixels,
                        persistentCorePixels);

                secondaryReview =
                    secondaryClean
                        ? "pass"
                        : "fail";

                Cv2.BitwiseOr(
                    secondaryCoreDebug,
                    originalCoreMask,
                    secondaryCoreDebug);

                Cv2.BitwiseOr(
                    secondaryLinkedDebug,
                    coreLinkedResidual,
                    secondaryLinkedDebug);

                Cv2.BitwiseOr(
                    secondaryPersistentDebug,
                    persistentCoreMask,
                    secondaryPersistentDebug);
            }

            var reviewDecision =
                ResolveReviewDecision(
                    primaryClean,
                    secondaryClean);

            residualAfterTotal +=
                selectedEffectiveResidual;

            audits[i] =
                previous with
                {
                    CoreMaskPixels =
                        coreMaskPixels,
                    ResidualAfterRetryPixels =
                        remainingAfterRetry,
                    EffectiveResidualAfterRetryPixels =
                        effectiveRemainingAfterRetry,
                    CoreOverlapAfterRetryPixels =
                        coreOverlapPixels,
                    PersistentCoreAfterRetryPixels =
                        persistentCorePixels,
                    SelectedResidualPixels =
                        selectedEffectiveResidual,
                    SelectedPersistentCorePixels =
                        persistentCorePixels,
                    SelectedPersistenceRatio =
                        PersistenceRatio(
                            persistentCorePixels,
                            coreMaskPixels),
                    SelectedPass =
                        selectedPass,
                    PrimaryReview =
                        primaryClean
                            ? "pass"
                            : "fail",
                    SecondaryReview =
                        secondaryReview,
                    RescuedBySecondary =
                        reviewDecision.RescuedBySecondary,
                    ReviewReason =
                        previous.InitialMaskPixels < 2
                            ? "mask_empty"
                            : reviewDecision.Reason,
                    Status =
                        previous.InitialMaskPixels < 2
                            ? "mask_empty"
                            : reviewDecision.Clean
                                ? reviewDecision.RescuedBySecondary
                                    ? "clean_after_secondary"
                                    : selectedPass == "retry"
                                        ? "clean_after_retry"
                                        : "clean_after_first_pass"
                                : "review_required"
                };
        }

        // 0033: A flat background can still contain colored glyph fragments
        // just outside the accepted foreground mask. The 0032 quality gate
        // already detects those fragments; use the same evidence to build a
        // tightly constrained cleanup mask instead of merely preserving the
        // original. Cleanup is allowed only for FLAT_FILL targets that fail
        // the chromatic residual gate, and it always reuses the stored flat
        // background color.
        using var flatResidualOutlierDebugMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var flatCleanupDebugMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

        using var preFlatCleanup =
            finalCleaned.Clone();

        var flatChromaticResidualBeforeCleanupByTarget =
            new Dictionary<string, (int ReviewPixels, int OutlierPixels, double Ratio)>(
                StringComparer.Ordinal);

        var flatCleanupPixelsByTarget =
            new Dictionary<string, int>(
                StringComparer.Ordinal);

        int flatResidualCleanupTargetCount = 0;

        foreach (var target in targets)
        {
            if (!backgroundAuditByTarget.TryGetValue(
                    target.TextRegionId,
                    out var backgroundAudit) ||
                !backgroundAudit.FlatAccepted)
            {
                continue;
            }

            using var originalTargetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            var beforeCleanup =
                MeasureFlatChromaticResidual(
                    preFlatCleanup,
                    originalTargetMask,
                    target.TextBounds,
                    backgroundAudit);

            flatChromaticResidualBeforeCleanupByTarget[
                target.TextRegionId] =
                beforeCleanup;

            using var chromaticOutlierMask =
                BuildFlatChromaticOutlierMask(
                    preFlatCleanup,
                    originalTargetMask,
                    target.TextBounds,
                    backgroundAudit);

            Cv2.BitwiseOr(
                flatResidualOutlierDebugMask,
                chromaticOutlierMask,
                flatResidualOutlierDebugMask);

            if (!IsFlatBackgroundQualityFailure(
                    beforeCleanup.ReviewPixels,
                    beforeCleanup.OutlierPixels,
                    beforeCleanup.Ratio))
            {
                flatCleanupPixelsByTarget[
                    target.TextRegionId] = 0;

                continue;
            }

            using var cleanupMask =
                BuildFlatResidualCleanupMask(
                    preFlatCleanup,
                    originalTargetMask,
                    target.TextBounds,
                    backgroundAudit);

            int cleanupPixels =
                Cv2.CountNonZero(
                    cleanupMask);

            flatCleanupPixelsByTarget[
                target.TextRegionId] =
                cleanupPixels;

            if (cleanupPixels < 2)
            {
                continue;
            }

            BackgroundReconstructionV2.ApplyFlatFill(
                finalCleaned,
                cleanupMask,
                backgroundAudit);

            Cv2.BitwiseOr(
                flatCleanupDebugMask,
                cleanupMask,
                flatCleanupDebugMask);

            flatResidualCleanupTargetCount++;
        }

        SaveColoredMaskDebug(
            preFlatCleanup,
            flatResidualOutlierDebugMask,
            new Scalar(0, 0, 255),
            flatResidualOutlierPath);

        SaveColoredMaskDebug(
            preFlatCleanup,
            flatCleanupDebugMask,
            new Scalar(255, 255, 0),
            flatCleanupMaskPath);

        var flatChromaticResidualByTarget =
            new Dictionary<string, (int ReviewPixels, int OutlierPixels, double Ratio)>(
                StringComparer.Ordinal);

        foreach (var target in targets)
        {
            if (!backgroundAuditByTarget.TryGetValue(
                    target.TextRegionId,
                    out var backgroundAudit) ||
                !backgroundAudit.FlatAccepted)
            {
                continue;
            }

            using var originalTargetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            flatChromaticResidualByTarget[
                target.TextRegionId] =
                MeasureFlatChromaticResidual(
                    finalCleaned,
                    originalTargetMask,
                    target.TextBounds,
                    backgroundAudit);
        }

        var backgroundQualityFailedTextRegionIds =
            flatChromaticResidualByTarget
                .Where(x =>
                    IsFlatBackgroundQualityFailure(
                        x.Value.ReviewPixels,
                        x.Value.OutlierPixels,
                        x.Value.Ratio))
                .Select(x =>
                    x.Key)
                .OrderBy(x => x)
                .ToArray();

        SaveColoredMaskDebug(
            source,
            secondaryCoreDebug,
            new Scalar(0, 255, 0),
            glyphCorePath);

        SaveColoredMaskDebug(
            finalCleaned,
            secondaryLinkedDebug,
            new Scalar(0, 255, 255),
            coreLinkedResidualPath);

        SaveColoredMaskDebug(
            finalCleaned,
            secondaryPersistentDebug,
            new Scalar(0, 0, 255),
            persistentCorePath);

        SaveLosslessWebp(
            cleanedPath,
            finalCleaned);

        var checkpointAudits =
            BuildCleanedCheckpointAudits(
                snapshot,
                targets,
                audits);

        SaveCleanedCheckpointDebug(
            finalCleaned,
            snapshot,
            checkpointAudits,
            legacyCheckpointPath);

        var detectedBubbleIds =
            snapshot.RawRegions
                .Where(x =>
                    x.Kind ==
                    LocalMangaTranslator.Models.PageRegionKind.Bubble)
                .Select(x =>
                    x.RegionId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var eraseTargetBubbleIds =
            checkpointAudits
                .Where(x =>
                    x.Kind == "Bubble")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var emptyVerifiedBubbleIds =
            checkpointAudits
                .Where(x =>
                    x.Kind == "Bubble" &&
                    x.EmptyVerified)
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var eraseTargetTextFreeIds =
            checkpointAudits
                .Where(x =>
                    x.Kind == "TextFree")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var emptyVerifiedTextFreeIds =
            checkpointAudits
                .Where(x =>
                    x.Kind == "TextFree" &&
                    x.EmptyVerified)
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var checkpointMissingBubbleIds =
            eraseTargetBubbleIds
                .Except(
                    emptyVerifiedBubbleIds,
                    StringComparer.Ordinal)
                .OrderBy(x => x)
                .ToArray();

        var checkpointMissingTextFreeIds =
            eraseTargetTextFreeIds
                .Except(
                    emptyVerifiedTextFreeIds,
                    StringComparer.Ordinal)
                .OrderBy(x => x)
                .ToArray();

        bool bubbleCheckpointPass =
            CheckpointIdsMatch(
                eraseTargetBubbleIds,
                emptyVerifiedBubbleIds);

        bool textFreeCheckpointPass =
            CheckpointIdsMatch(
                eraseTargetTextFreeIds,
                emptyVerifiedTextFreeIds);

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
                GrayscaleMaskPixels =
                    Cv2.CountNonZero(
                        grayscaleMask),
                ColorRescueMaskPixels =
                    Cv2.CountNonZero(
                        colorRescueMask),
                ResidualBeforeRetryPixels =
                    audits.Sum(x =>
                        x.ResidualBeforeRetryPixels),
                RetryMaskPixels =
                    retryMaskPixels,
                RetryReconstruction =
                    new
                    {
                        FlatFillRetryTargets =
                            flatRetryTargetCount,
                        FlatFillRetryMaskPixels =
                            flatRetryMaskPixels,
                        TeleaRetryTargets =
                            teleaRetryTargetCount,
                        TeleaRetryMaskPixels =
                            teleaRetryMaskPixels
                    },
                FlatResidualCleanup =
                    new
                    {
                        AppliedTargets =
                            flatResidualCleanupTargetCount,
                        CleanupMaskPixels =
                            flatCleanupPixelsByTarget.Values.Sum(),
                        OutlierDebugFile =
                            Path.GetFileName(
                                flatResidualOutlierPath),
                        CleanupMaskDebugFile =
                            Path.GetFileName(
                                flatCleanupMaskPath)
                    },
                ResidualAfterRetryPixels =
                    residualAfterTotal,
                RetryTargetCount =
                    retryTargetCount,
                Clean =
                    audits.All(IsAuditClean),
                BackgroundReconstruction =
                    new
                    {
                        FlatFillTargets =
                            backgroundAudits.Count(x =>
                                x.FlatAccepted),
                        TeleaTargets =
                            backgroundAudits.Count(x =>
                                !x.FlatAccepted),
                        BackgroundQualityFailures =
                            backgroundQualityFailedTextRegionIds.Length,
                        BackgroundQualityFailedTextRegionIds =
                            backgroundQualityFailedTextRegionIds,
                        Items =
                            backgroundAudits.Select(x =>
                                new
                                {
                                    x.TextRegionId,
                                    x.Kind,
                                    x.Strategy,
                                    x.MaskPixels,
                                    x.SampleCount,
                                    Background =
                                        new
                                        {
                                            B = x.BackgroundB,
                                            G = x.BackgroundG,
                                            R = x.BackgroundR
                                        },
                                    x.DominantMatchRatio,
                                    x.P75ColorDistance,
                                    x.P90ColorDistance,

                                    x.SpatialCoverage,

                                    x.ExclusionRadius,
                                    x.FlatAccepted,
                                    x.Reason,
                                    PostReconstructionP85 =
                                        backgroundQualityP85.GetValueOrDefault(
                                            x.TextRegionId,
                                            -1),
                                    // 0032-compatible aliases describe the state
                                    // immediately after retry, before 0033 cleanup.
                                    PostRetryChromaticReviewPixels =
                                        flatChromaticResidualBeforeCleanupByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticResidualBefore)
                                            ? chromaticResidualBefore.ReviewPixels
                                            : 0,
                                    PostRetryChromaticOutlierPixels =
                                        flatChromaticResidualBeforeCleanupByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticOutlierBefore)
                                            ? chromaticOutlierBefore.OutlierPixels
                                            : 0,
                                    PostRetryChromaticOutlierRatio =
                                        flatChromaticResidualBeforeCleanupByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticRatioBefore)
                                            ? chromaticRatioBefore.Ratio
                                            : 0,
                                    FlatResidualCleanupApplied =
                                        flatCleanupPixelsByTarget.GetValueOrDefault(
                                            x.TextRegionId) > 0,
                                    FlatResidualCleanupMaskPixels =
                                        flatCleanupPixelsByTarget.GetValueOrDefault(
                                            x.TextRegionId),
                                    PostCleanupChromaticReviewPixels =
                                        flatChromaticResidualByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticResidualAfter)
                                            ? chromaticResidualAfter.ReviewPixels
                                            : 0,
                                    PostCleanupChromaticOutlierPixels =
                                        flatChromaticResidualByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticOutlierAfter)
                                            ? chromaticOutlierAfter.OutlierPixels
                                            : 0,
                                    PostCleanupChromaticOutlierRatio =
                                        flatChromaticResidualByTarget.TryGetValue(
                                            x.TextRegionId,
                                            out var chromaticRatioAfter)
                                            ? chromaticRatioAfter.Ratio
                                            : 0,
                                    BackgroundQualityPass =
                                        !backgroundQualityFailedTextRegionIds.Contains(
                                            x.TextRegionId,
                                            StringComparer.Ordinal)
                                })
                    },
                LegacyReviewerCheckpoint =
                    new
                    {
                        DetectedBubbleCount =
                            detectedBubbleIds.Count,
                        EraseTargetBubbleCount =
                            eraseTargetBubbleIds.Count,
                        EmptyVerifiedBubbleCount =
                            emptyVerifiedBubbleIds.Count,
                        BubbleCheckpointPass =
                            bubbleCheckpointPass,
                        EraseTargetBubbleIds =
                            eraseTargetBubbleIds
                                .OrderBy(x => x)
                                .ToArray(),
                        EmptyVerifiedBubbleIds =
                            emptyVerifiedBubbleIds
                                .OrderBy(x => x)
                                .ToArray(),
                        CheckpointMissingBubbleIds =
                            checkpointMissingBubbleIds,
                        EraseTargetTextFreeCount =
                            eraseTargetTextFreeIds.Count,
                        EmptyVerifiedTextFreeCount =
                            emptyVerifiedTextFreeIds.Count,
                        TextFreeCheckpointPass =
                            textFreeCheckpointPass,
                        EraseTargetTextFreeIds =
                            eraseTargetTextFreeIds
                                .OrderBy(x => x)
                                .ToArray(),
                        EmptyVerifiedTextFreeIds =
                            emptyVerifiedTextFreeIds
                                .OrderBy(x => x)
                                .ToArray(),
                        CheckpointMissingTextFreeIds =
                            checkpointMissingTextFreeIds,
                        OverallPass =
                            bubbleCheckpointPass &&
                            textFreeCheckpointPass,
                        Items =
                            checkpointAudits.Select(x =>
                                new
                                {
                                    x.CheckpointId,
                                    x.Kind,
                                    Bounds =
                                        new
                                        {
                                            x.Bounds.X,
                                            x.Bounds.Y,
                                            x.Bounds.Width,
                                            x.Bounds.Height
                                        },
                                    x.TextRegionIds,
                                    x.EmptyVerified,
                                    x.Status
                                })
                    },
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
                            GrayscaleMaskPixels =
                                grayscalePixelsByTarget.GetValueOrDefault(
                                    target.TextRegionId),
                            ColorRescueMaskPixels =
                                colorPixelsByTarget.GetValueOrDefault(
                                    target.TextRegionId),
                            audit.InitialMaskPixels,
                            audit.CoreMaskPixels,
                            audit.ResidualBeforeRetryPixels,
                            audit.EffectiveResidualBeforeRetryPixels,
                            audit.CoreOverlapBeforeRetryPixels,
                            audit.PersistentCoreBeforeRetryPixels,
                            audit.RetryMaskPixels,
                            RetryStrategy =
                                audit.Retried
                                    ? backgroundAuditByTarget.TryGetValue(
                                          target.TextRegionId,
                                          out var retryBackgroundAudit)
                                        ? retryBackgroundAudit.FlatAccepted
                                            ? "FLAT_FILL"
                                            : "TELEA"
                                        : "TELEA_FALLBACK"
                                    : "not_run",
                            audit.ResidualAfterRetryPixels,
                            audit.EffectiveResidualAfterRetryPixels,
                            audit.CoreOverlapAfterRetryPixels,
                            audit.PersistentCoreAfterRetryPixels,
                            audit.SelectedResidualPixels,
                            audit.SelectedPersistentCorePixels,
                            audit.SelectedPersistenceRatio,
                            audit.SelectedPass,
                            audit.Retried,
                            audit.PrimaryReview,
                            audit.SecondaryReview,
                            audit.RescuedBySecondary,
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
            legacyCheckpointPath,
            jsonPath,
            audits.ToArray(),
            checkpointAudits,
            backgroundQualityFailedTextRegionIds);
    }

    public static bool CheckpointIdsMatch(
        IEnumerable<string> expectedIds,
        IEnumerable<string> verifiedIds)
    {
        var expected =
            expectedIds.ToHashSet(
                StringComparer.Ordinal);

        var verified =
            verifiedIds.ToHashSet(
                StringComparer.Ordinal);

        return expected.SetEquals(
            verified);
    }

    static IReadOnlyList<V2CleanedCheckpointAudit>
        BuildCleanedCheckpointAudits(
            V2DetectionSnapshot snapshot,
            IReadOnlyList<V2TextTarget> targets,
            IReadOnlyList<V2EraseTargetAudit> audits)
    {
        var auditByTarget =
            audits.ToDictionary(
                x => x.TextRegionId,
                StringComparer.Ordinal);

        var items =
            new List<V2CleanedCheckpointAudit>();

        foreach (var group in targets
                     .Where(x =>
                         x.BubbleRegionId is not null)
                     .GroupBy(
                         x => x.BubbleRegionId!,
                         StringComparer.Ordinal))
        {
            var ids =
                group.Select(x =>
                        x.TextRegionId)
                    .Distinct(
                        StringComparer.Ordinal)
                    .OrderBy(x => x)
                    .ToArray();

            bool verified =
                ids.Length > 0 &&
                ids.All(id =>
                    auditByTarget.TryGetValue(
                        id,
                        out var audit) &&
                    IsAuditClean(
                        audit));

            var bubbleBounds =
                group.Select(x =>
                        x.BubbleBounds)
                    .FirstOrDefault(x =>
                        x.HasValue);

            Rect bounds =
                bubbleBounds ??
                UnionBounds(
                    group.Select(x =>
                        x.TextBounds));

            items.Add(
                new V2CleanedCheckpointAudit(
                    group.Key,
                    "Bubble",
                    bounds,
                    ids,
                    verified,
                    verified
                        ? "EMPTY_OK"
                        : "ERASE_CHECK"));
        }

        foreach (var target in targets.Where(x =>
                     x.Kind ==
                     LocalMangaTranslator.Models.PageRegionKind.TextFree))
        {
            bool verified =
                auditByTarget.TryGetValue(
                    target.TextRegionId,
                    out var audit) &&
                IsAuditClean(
                    audit);

            items.Add(
                new V2CleanedCheckpointAudit(
                    target.TextRegionId,
                    "TextFree",
                    target.TextBounds,
                    [target.TextRegionId],
                    verified,
                    verified
                        ? "EMPTY_OK"
                        : "ERASE_CHECK"));
        }

        foreach (var target in targets.Where(x =>
                     x.Kind ==
                         LocalMangaTranslator.Models.PageRegionKind.TextBubble &&
                     x.BubbleRegionId is null))
        {
            bool verified =
                auditByTarget.TryGetValue(
                    target.TextRegionId,
                    out var audit) &&
                IsAuditClean(
                    audit);

            items.Add(
                new V2CleanedCheckpointAudit(
                    target.TextRegionId,
                    "UnparentedTextBubble",
                    target.TextBounds,
                    [target.TextRegionId],
                    verified,
                    verified
                        ? "EMPTY_OK"
                        : "ERASE_CHECK"));
        }

        return items
            .OrderBy(x =>
                x.Bounds.Y)
            .ThenBy(x =>
                x.Bounds.X)
            .ToArray();
    }

    static Rect UnionBounds(
        IEnumerable<Rect> bounds)
    {
        var all =
            bounds.ToArray();

        if (all.Length == 0)
            return default;

        int left =
            all.Min(x =>
                x.Left);

        int top =
            all.Min(x =>
                x.Top);

        int right =
            all.Max(x =>
                x.Right);

        int bottom =
            all.Max(x =>
                x.Bottom);

        return new Rect(
            left,
            top,
            Math.Max(
                0,
                right - left),
            Math.Max(
                0,
                bottom - top));
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

    public static (
        bool Clean,
        bool SecondaryRan,
        bool RescuedBySecondary,
        string Reason)
        ResolveReviewDecision(
            bool primaryClean,
            bool secondaryClean)
    {
        if (primaryClean)
        {
            return (
                true,
                false,
                false,
                "primary_residual_clean");
        }

        if (secondaryClean)
        {
            return (
                true,
                true,
                true,
                "secondary_glyph_rescue");
        }

        return (
            false,
            true,
            false,
            "original_glyph_persistence");
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
        using var persistent =
            BuildOriginalGlyphPersistenceMask(
                source,
                cleaned,
                originalCoreMask,
                coreLinkedResidual,
                textBounds);

        return Cv2.CountNonZero(
            persistent);
    }

    public static Mat BuildOriginalGlyphPersistenceMask(
        Mat source,
        Mat cleaned,
        Mat originalCoreMask,
        Mat coreLinkedResidual,
        Rect textBounds)
    {
        var persistentMask =
            Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1)
            .ToMat();

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
                    persistentMask.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        return persistentMask;
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

            // A bubble border or artwork line may enter the review halo.
            // It must place a meaningful share of its own component on the
            // immutable glyph core; a large border that only brushes the core
            // is not surviving source text.
            if (componentOverlapRatio < 0.08)
                continue;

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

    static Mat ReconstructFirstPass(
        Mat source,
        IReadOnlyList<V2TextTarget> targets,
        out IReadOnlyList<V2BackgroundReconstructionAudit> audits,
        out IReadOnlyDictionary<string, double> qualityP85)
    {
        using var working =
            source.Clone();

        using var teleaMask =
            Mat.Zeros(
                    source.Rows,
                    source.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        var auditList =
            new List<V2BackgroundReconstructionAudit>(
                targets.Count);

        foreach (var target in targets)
        {
            using var targetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            var audit =
                BackgroundReconstructionV2.Analyze(
                    source,
                    target,
                    targetMask);

            auditList.Add(
                audit);

            if (audit.FlatAccepted)
            {
                BackgroundReconstructionV2.ApplyFlatFill(
                    working,
                    targetMask,
                    audit);
            }
            else
            {
                Cv2.BitwiseOr(
                    teleaMask,
                    targetMask,
                    teleaMask);
            }
        }

        Mat reconstructed;

        if (Cv2.CountNonZero(
                teleaMask) > 0)
        {
            reconstructed =
                InpaintOnlyMaskedPixels(
                    working,
                    teleaMask);
        }
        else
        {
            reconstructed =
                working.Clone();
        }

        var quality =
            new Dictionary<string, double>(
                StringComparer.Ordinal);

        foreach (var target in targets)
        {
            var audit =
                auditList.First(x =>
                    string.Equals(
                        x.TextRegionId,
                        target.TextRegionId,
                        StringComparison.Ordinal));

            if (!audit.FlatAccepted)
            {
                quality[
                    target.TextRegionId] =
                    -1;

                continue;
            }

            using var targetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            quality[
                target.TextRegionId] =
                BackgroundReconstructionV2
                    .MeasureFilledBackgroundP85(
                        reconstructed,
                        targetMask,
                        audit);
        }

        audits =
            auditList;

        qualityP85 =
            quality;

        return reconstructed;
    }

    static void SaveBackgroundStrategyDebug(
        Mat reconstructed,
        IReadOnlyList<V2BackgroundReconstructionAudit> audits,
        IReadOnlyDictionary<string, double> qualityP85,
        string path)
    {
        using var debug =
            reconstructed.Clone();

        foreach (var audit in audits)
        {
            Scalar color =
                audit.FlatAccepted
                    ? new Scalar(
                        0,
                        220,
                        0)
                    : new Scalar(
                        0,
                        165,
                        255);

            Cv2.Rectangle(
                debug,
                audit.Bounds,
                color,
                2);

            double quality =
                qualityP85.GetValueOrDefault(
                    audit.TextRegionId,
                    -1);

            string label =
                audit.FlatAccepted
                    ? $"{audit.TextRegionId} FLAT_FILL bg={audit.BackgroundR},{audit.BackgroundG},{audit.BackgroundB} q={quality:0.0}"
                    : $"{audit.TextRegionId} TELEA";

            Cv2.PutText(
                debug,
                label,
                new Point(
                    Math.Max(
                        2,
                        audit.Bounds.X),
                    Math.Max(
                        16,
                        audit.Bounds.Y - 5)),
                HersheyFonts.HersheySimplex,
                0.42,
                color,
                1,
                LineTypes.AntiAlias);
        }

        int flat =
            audits.Count(x =>
                x.FlatAccepted);

        int telea =
            audits.Count -
            flat;

        int failed =
            audits.Count(x =>
                x.FlatAccepted &&
                qualityP85.GetValueOrDefault(
                    x.TextRegionId,
                    double.MaxValue) >
                18.0);

        string summary =
            $"Background reconstruction | FLAT_FILL {flat} | TELEA {telea} | quality fail {failed}";

        Cv2.Rectangle(
            debug,
            new Rect(
                0,
                0,
                Math.Min(
                    debug.Cols,
                    Math.Max(
                        520,
                        summary.Length * 10)),
                Math.Min(
                    debug.Rows,
                    34)),
            new Scalar(
                20,
                20,
                20),
            thickness: -1);

        Cv2.PutText(
            debug,
            summary,
            new Point(
                8,
                23),
            HersheyFonts.HersheySimplex,
            0.55,
            Scalar.White,
            1,
            LineTypes.AntiAlias);

        SaveLosslessWebp(
            path,
            debug);
    }

    static void WriteBackgroundAudit(
        string sourcePath,
        IReadOnlyList<V2BackgroundReconstructionAudit> audits,
        IReadOnlyDictionary<string, double> qualityP85,
        string path)
    {
        var document =
            new
            {
                Schema =
                    "pipeline-v2-background-reconstruction-v1",
                SourceFile =
                    Path.GetFileName(
                        sourcePath),
                TargetCount =
                    audits.Count,
                FlatFillTargets =
                    audits.Count(x =>
                        x.FlatAccepted),
                TeleaTargets =
                    audits.Count(x =>
                        !x.FlatAccepted),
                BackgroundQualityFailures =
                    audits.Count(x =>
                        x.FlatAccepted &&
                        qualityP85.GetValueOrDefault(
                            x.TextRegionId,
                            double.MaxValue) >
                        18.0),
                Items =
                    audits.Select(x =>
                        new
                        {
                            x.TextRegionId,
                            x.Kind,
                            Bounds =
                                new
                                {
                                    x.Bounds.X,
                                    x.Bounds.Y,
                                    x.Bounds.Width,
                                    x.Bounds.Height
                                },
                            x.Strategy,
                            x.MaskPixels,
                            x.SampleCount,
                            Background =
                                new
                                {
                                    B = x.BackgroundB,
                                    G = x.BackgroundG,
                                    R = x.BackgroundR
                                },
                            x.DominantMatchRatio,
                            x.P75ColorDistance,
                            x.P90ColorDistance,

                            x.SpatialCoverage,

                            x.ExclusionRadius,
                            x.FlatAccepted,
                            x.Reason,
                            PostReconstructionP85 =
                                qualityP85.GetValueOrDefault(
                                    x.TextRegionId,
                                    -1),
                            BackgroundQualityPass =
                                !x.FlatAccepted ||
                                qualityP85.GetValueOrDefault(
                                    x.TextRegionId,
                                    double.MaxValue) <=
                                18.0
                        })
            };

        File.WriteAllText(
            path,
            JsonSerializer.Serialize(
                document,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));
    }

    public static bool IsFlatBackgroundQualityFailure(
        int reviewPixels,
        int outlierPixels,
        double ratio)
        =>
            outlierPixels >=
                Math.Max(
                    16,
                    (int)Math.Ceiling(
                        reviewPixels *
                        0.015)) &&
            ratio >= 0.015;

    static Mat BuildFlatReviewZone(
        Mat image,
        Mat originalGlyphMask,
        Rect textBounds,
        V2BackgroundReconstructionAudit audit)
    {
        int radius =
            Math.Clamp(
                audit.ExclusionRadius + 3,
                6,
                12);

        using var expanded =
            new Mat();

        using (var kernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Ellipse,
                   new Size(
                       radius * 2 + 1,
                       radius * 2 + 1)))
        {
            Cv2.Dilate(
                originalGlyphMask,
                expanded,
                kernel,
                iterations: 1);
        }

        using var originalInverse =
            new Mat();

        Cv2.BitwiseNot(
            originalGlyphMask,
            originalInverse);

        var reviewZone =
            new Mat();

        Cv2.BitwiseAnd(
            expanded,
            originalInverse,
            reviewZone);

        using var boundsMask =
            BuildBoundsMask(
                image.Rows,
                image.Cols,
                textBounds);

        Cv2.BitwiseAnd(
            reviewZone,
            boundsMask,
            reviewZone);

        return reviewZone;
    }

    public static Mat BuildFlatChromaticOutlierMask(
        Mat image,
        Mat originalGlyphMask,
        Rect textBounds,
        V2BackgroundReconstructionAudit audit)
    {
        var outlierMask =
            Mat.Zeros(
                image.Rows,
                image.Cols,
                MatType.CV_8UC1)
            .ToMat();

        if (!audit.FlatAccepted)
        {
            return outlierMask;
        }

        using var reviewZone =
            BuildFlatReviewZone(
                image,
                originalGlyphMask,
                textBounds,
                audit);

        double expectedChroma =
            Math.Max(
                audit.BackgroundB,
                Math.Max(
                    audit.BackgroundG,
                    audit.BackgroundR)) -
            Math.Min(
                audit.BackgroundB,
                Math.Min(
                    audit.BackgroundG,
                    audit.BackgroundR));

        for (int y =
                 Math.Max(
                     0,
                     textBounds.Top);
             y <
             Math.Min(
                 image.Rows,
                 textBounds.Bottom);
             y++)
        {
            for (int x =
                     Math.Max(
                         0,
                         textBounds.Left);
                 x <
                 Math.Min(
                     image.Cols,
                     textBounds.Right);
                 x++)
            {
                if (reviewZone.At<byte>(
                        y,
                        x) == 0)
                {
                    continue;
                }

                var pixel =
                    image.At<Vec3b>(
                        y,
                        x);

                double db =
                    pixel.Item0 -
                    audit.BackgroundB;

                double dg =
                    pixel.Item1 -
                    audit.BackgroundG;

                double dr =
                    pixel.Item2 -
                    audit.BackgroundR;

                double distance =
                    Math.Sqrt(
                        db * db +
                        dg * dg +
                        dr * dr);

                int pixelMax =
                    Math.Max(
                        pixel.Item0,
                        Math.Max(
                            pixel.Item1,
                            pixel.Item2));

                int pixelMin =
                    Math.Min(
                        pixel.Item0,
                        Math.Min(
                            pixel.Item1,
                            pixel.Item2));

                double pixelChroma =
                    pixelMax -
                    pixelMin;

                if (distance >= 70.0 &&
                    pixelChroma -
                    expectedChroma >= 40.0)
                {
                    outlierMask.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        return outlierMask;
    }

    /// <summary>
    /// 0033 flat residual cleanup. Colored residual pixels are the seed. The
    /// mask may grow only through pixels that are still strongly inconsistent
    /// with the estimated flat background and only inside the immutable
    /// TextBounds / glyph-adjacent review zone. This lets a red/blue glyph seed
    /// pull in its attached neutral black outline without sweeping up an
    /// unrelated bubble border.
    /// </summary>
    public static Mat BuildFlatResidualCleanupMask(
        Mat image,
        Mat originalGlyphMask,
        Rect textBounds,
        V2BackgroundReconstructionAudit audit)
    {
        var empty =
            Mat.Zeros(
                image.Rows,
                image.Cols,
                MatType.CV_8UC1)
            .ToMat();

        if (!audit.FlatAccepted)
        {
            return empty;
        }

        using var reviewZone =
            BuildFlatReviewZone(
                image,
                originalGlyphMask,
                textBounds,
                audit);

        using var seed =
            BuildFlatChromaticOutlierMask(
                image,
                originalGlyphMask,
                textBounds,
                audit);

        if (Cv2.CountNonZero(
                seed) < 1)
        {
            return empty;
        }

        using var candidate =
            Mat.Zeros(
                image.Rows,
                image.Cols,
                MatType.CV_8UC1)
            .ToMat();

        double candidateDistanceThreshold =
            Math.Clamp(
                audit.P90ColorDistance + 24.0,
                42.0,
                60.0);

        for (int y =
                 Math.Max(
                     0,
                     textBounds.Top);
             y <
             Math.Min(
                 image.Rows,
                 textBounds.Bottom);
             y++)
        {
            for (int x =
                     Math.Max(
                         0,
                         textBounds.Left);
                 x <
                 Math.Min(
                     image.Cols,
                     textBounds.Right);
                 x++)
            {
                if (reviewZone.At<byte>(
                        y,
                        x) == 0)
                {
                    continue;
                }

                var pixel =
                    image.At<Vec3b>(
                        y,
                        x);

                double db =
                    pixel.Item0 -
                    audit.BackgroundB;

                double dg =
                    pixel.Item1 -
                    audit.BackgroundG;

                double dr =
                    pixel.Item2 -
                    audit.BackgroundR;

                double distance =
                    Math.Sqrt(
                        db * db +
                        dg * dg +
                        dr * dr);

                if (distance >=
                    candidateDistanceThreshold)
                {
                    candidate.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        using var kernel =
            Cv2.GetStructuringElement(
                MorphShapes.Rect,
                new Size(3, 3));

        var linked =
            seed.Clone();

        int growthSteps =
            Math.Clamp(
                audit.ExclusionRadius + 3,
                6,
                12);

        for (int i = 0;
             i < growthSteps;
             i++)
        {
            using var dilated =
                new Mat();

            Cv2.Dilate(
                linked,
                dilated,
                kernel,
                iterations: 1);

            Cv2.BitwiseAnd(
                dilated,
                candidate,
                dilated);

            Cv2.BitwiseOr(
                linked,
                dilated,
                linked);
        }

        empty.Dispose();

        return linked;
    }

    static (int ReviewPixels, int OutlierPixels, double Ratio)
        MeasureFlatChromaticResidual(
            Mat image,
            Mat originalGlyphMask,
            Rect textBounds,
            V2BackgroundReconstructionAudit audit)
    {
        using var reviewZone =
            BuildFlatReviewZone(
                image,
                originalGlyphMask,
                textBounds,
                audit);

        using var outlierMask =
            BuildFlatChromaticOutlierMask(
                image,
                originalGlyphMask,
                textBounds,
                audit);

        int reviewPixels =
            Cv2.CountNonZero(
                reviewZone);

        int outlierPixels =
            Cv2.CountNonZero(
                outlierMask);

        double ratio =
            reviewPixels == 0
                ? 0
                : outlierPixels /
                  (double)reviewPixels;

        return (
            reviewPixels,
            outlierPixels,
            ratio);
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

    static void SaveCleanedCheckpointDebug(
        Mat cleaned,
        V2DetectionSnapshot snapshot,
        IReadOnlyList<V2CleanedCheckpointAudit> checkpoints,
        string path)
    {
        using var debug =
            cleaned.Clone();

        var selectedBubbleIds =
            checkpoints
                .Where(x =>
                    x.Kind == "Bubble")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        foreach (var region in snapshot.RawRegions.Where(x =>
                     x.Kind ==
                     LocalMangaTranslator.Models.PageRegionKind.Bubble &&
                     !selectedBubbleIds.Contains(
                         x.RegionId)))
        {
            Cv2.Rectangle(
                debug,
                region.Bounds,
                new Scalar(
                    110,
                    110,
                    110),
                1);
        }

        foreach (var checkpoint in checkpoints)
        {
            Scalar color =
                checkpoint.EmptyVerified
                    ? new Scalar(
                        0,
                        220,
                        0)
                    : new Scalar(
                        0,
                        0,
                        255);

            Cv2.Rectangle(
                debug,
                checkpoint.Bounds,
                color,
                3);

            string label =
                $"{checkpoint.CheckpointId} {checkpoint.Status}";

            int baseline;
            var labelSize =
                Cv2.GetTextSize(
                    label,
                    HersheyFonts.HersheySimplex,
                    0.45,
                    1,
                    out baseline);

            int labelX =
                Math.Clamp(
                    checkpoint.Bounds.X,
                    0,
                    Math.Max(
                        0,
                        debug.Cols -
                        labelSize.Width -
                        6));

            int labelY =
                Math.Clamp(
                    checkpoint.Bounds.Y - 5,
                    labelSize.Height + 4,
                    debug.Rows - 2);

            var labelRect =
                new Rect(
                    labelX,
                    labelY -
                    labelSize.Height -
                    4,
                    Math.Min(
                        debug.Cols - labelX,
                        labelSize.Width + 6),
                    Math.Min(
                        debug.Rows -
                        (labelY -
                         labelSize.Height -
                         4),
                        labelSize.Height +
                        baseline +
                        6));

            if (labelRect.Width > 0 &&
                labelRect.Height > 0)
            {
                Cv2.Rectangle(
                    debug,
                    labelRect,
                    new Scalar(
                        20,
                        20,
                        20),
                    thickness: -1);
            }

            Cv2.PutText(
                debug,
                label,
                new Point(
                    labelX + 3,
                    labelY),
                HersheyFonts.HersheySimplex,
                0.45,
                color,
                1,
                LineTypes.AntiAlias);
        }

        int eraseTargetBubbles =
            checkpoints.Count(x =>
                x.Kind == "Bubble");

        int verifiedBubbles =
            checkpoints.Count(x =>
                x.Kind == "Bubble" &&
                x.EmptyVerified);

        int eraseTargetTextFree =
            checkpoints.Count(x =>
                x.Kind == "TextFree");

        int verifiedTextFree =
            checkpoints.Count(x =>
                x.Kind == "TextFree" &&
                x.EmptyVerified);

        string summary =
            $"Bubble {verifiedBubbles}/{eraseTargetBubbles} EMPTY  |  " +
            $"TextFree {verifiedTextFree}/{eraseTargetTextFree} EMPTY";

        int summaryBaseline;
        var summarySize =
            Cv2.GetTextSize(
                summary,
                HersheyFonts.HersheySimplex,
                0.6,
                2,
                out summaryBaseline);

        int summaryWidth =
            Math.Min(
                debug.Cols,
                summarySize.Width + 18);

        Cv2.Rectangle(
            debug,
            new Rect(
                0,
                0,
                summaryWidth,
                Math.Min(
                    debug.Rows,
                    summarySize.Height +
                    summaryBaseline +
                    14)),
            new Scalar(
                20,
                20,
                20),
            thickness: -1);

        Cv2.PutText(
            debug,
            summary,
            new Point(
                8,
                summarySize.Height + 6),
            HersheyFonts.HersheySimplex,
            0.6,
            new Scalar(
                255,
                255,
                255),
            2,
            LineTypes.AntiAlias);

        SaveLosslessWebp(
            path,
            debug);
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

    static void SaveColoredMaskDebug(
        Mat baseImage,
        Mat mask,
        Scalar color,
        string path)
    {
        using var debug =
            baseImage.Clone();

        using var overlay =
            new Mat(
                baseImage.Rows,
                baseImage.Cols,
                MatType.CV_8UC3,
                color);

        overlay.CopyTo(
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
