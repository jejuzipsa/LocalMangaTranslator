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
    string CleanedCheckpointDebugPath,
    string DetectionJsonPath,
    IReadOnlyList<V2EraseTargetAudit> TargetAudits,
    IReadOnlyList<V2CleanedCheckpointAudit> CleanedCheckpointAudits);

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

        string firstCleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_02_cleaned_first.webp");

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

        string cleanedPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04_cleaned_final.webp");

        string cleanedCheckpointPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04b_cleaned_checkpoint.webp");

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

        using var finalCleaned =
            retryMaskPixels > 0
                ? InpaintOnlyMaskedPixels(
                    firstCleaned,
                    retryMask)
                : firstCleaned.Clone();

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
            cleanedCheckpointPath);

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
                CleanedCheckpoint =
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
            cleanedCheckpointPath,
            jsonPath,
            audits.ToArray(),
            checkpointAudits);
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
