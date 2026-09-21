using System.Text.Json;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record V2CleanedTargetVerification(
    string TextRegionId,
    Rect OriginalBounds,
    bool EmptyVerified,
    string Status,
    IReadOnlyList<string> ResidualRegionIds,
    float MaxResidualScore,
    bool LegacyReviewerClean);

public sealed record V2CleanedStateCheckpoint(
    string CheckpointId,
    string Kind,
    Rect Bounds,
    IReadOnlyList<string> TextRegionIds,
    bool EmptyVerified,
    string Status,
    bool LegacyReviewerClean);

public sealed record V2CleanedStateResult(
    bool VerifierAvailable,
    string VerifierMode,
    int ResidualTextDetectionCount,
    IReadOnlyList<V2CleanedTargetVerification> Targets,
    IReadOnlyList<V2CleanedStateCheckpoint> Checkpoints,
    string DebugPath,
    string JsonPath)
{
    public bool IsTargetEmpty(
        string textRegionId)
        => Targets.Any(x =>
            x.EmptyVerified &&
            string.Equals(
                x.TextRegionId,
                textRegionId,
                StringComparison.Ordinal));
}

/// <summary>
/// 0028 independent cleaned-state verifier.
///
/// The erase reviewer intentionally remains diagnostic evidence. This verifier
/// asks a different question on the already-cleaned image: does the comic
/// detector still see text inside the immutable source text target?
///
/// Original detector IDs and geometry stay authoritative. The cleaned image is
/// re-detected only for semantic residual-text evidence; it cannot create or
/// move the original erase targets.
/// </summary>
public sealed class CleanedStateVerifier
{
    const int LosslessWebpQuality = 101;

    static readonly object DetectorSync =
        new();

    static readonly RtdetrPageRegionAnalyzer Detector =
        new();

    public V2CleanedStateResult Verify(
        string sourcePath,
        string cleanedPath,
        string outputDirectory,
        V2DetectionSnapshot snapshot,
        IReadOnlySet<string> selectedTextRegionIds,
        IReadOnlyList<V2EraseTargetAudit> legacyAudits,
        CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested();

        string debugDir =
            OutputDirectoryLayout.Debug(
                outputDirectory);

        Directory.CreateDirectory(
            debugDir);

        string name =
            Path.GetFileNameWithoutExtension(
                sourcePath);

        string debugPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_04b_cleaned_checkpoint.webp");

        string jsonPath =
            Path.Combine(
                debugDir,
                $"{name}.v2_cleaned_state_audit.json");

        var selectedTargets =
            snapshot.TextTargets
                .Where(x =>
                    selectedTextRegionIds.Contains(
                        x.TextRegionId))
                .ToList();

        var legacyByTarget =
            legacyAudits.ToDictionary(
                x => x.TextRegionId,
                StringComparer.Ordinal);

        bool verifierAvailable =
            Detector.IsReady;

        IReadOnlyList<PageRegion> cleanedRegions =
            [];

        string verifierMode;

        if (verifierAvailable)
        {
            try
            {
                lock (DetectorSync)
                {
                    cleanedRegions =
                        Detector.Analyze(
                            cleanedPath,
                            token);
                }

                verifierMode =
                    "cleaned_rtdetr_text_redetection";
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                verifierAvailable =
                    false;

                verifierMode =
                    "legacy_reviewer_fallback";
            }
        }
        else
        {
            verifierMode =
                "legacy_reviewer_fallback";
        }

        var residualTextRegions =
            cleanedRegions
                .Where(x =>
                    x.Kind is
                        PageRegionKind.TextBubble or
                        PageRegionKind.TextFree)
                .ToList();

        var targetChecks =
            selectedTargets
                .Select(target =>
                {
                    bool legacyClean =
                        legacyByTarget.TryGetValue(
                            target.TextRegionId,
                            out var legacy) &&
                        ErasePipelineV2.IsAuditClean(
                            legacy);

                    if (!verifierAvailable)
                    {
                        return new V2CleanedTargetVerification(
                            target.TextRegionId,
                            target.TextBounds,
                            legacyClean,
                            legacyClean
                                ? "EMPTY_OK_LEGACY_FALLBACK"
                                : "ERASE_CHECK_LEGACY_FALLBACK",
                            [],
                            0,
                            legacyClean);
                    }

                    var matches =
                        residualTextRegions
                            .Where(candidate =>
                                IsResidualMatch(
                                    target.TextBounds,
                                    candidate.Bounds))
                            .OrderByDescending(x =>
                                x.Score)
                            .ToList();

                    bool empty =
                        matches.Count == 0;

                    return new V2CleanedTargetVerification(
                        target.TextRegionId,
                        target.TextBounds,
                        empty,
                        empty
                            ? "EMPTY_OK"
                            : "TEXT_REDETECTED",
                        matches
                            .Select(x =>
                                x.RegionId)
                            .ToArray(),
                        matches.Count == 0
                            ? 0
                            : matches.Max(x =>
                                x.Score),
                        legacyClean);
                })
                .ToArray();

        var targetById =
            targetChecks.ToDictionary(
                x => x.TextRegionId,
                StringComparer.Ordinal);

        var checkpoints =
            BuildCheckpoints(
                selectedTargets,
                targetById)
            .ToArray();

        SaveDebug(
            cleanedPath,
            snapshot,
            checkpoints,
            residualTextRegions,
            debugPath);

        WriteAudit(
            sourcePath,
            verifierAvailable,
            verifierMode,
            snapshot,
            selectedTargets,
            targetChecks,
            checkpoints,
            residualTextRegions,
            jsonPath);

        return new V2CleanedStateResult(
            verifierAvailable,
            verifierMode,
            residualTextRegions.Count,
            targetChecks,
            checkpoints,
            debugPath,
            jsonPath);
    }

    public static bool IsResidualMatch(
        Rect originalTarget,
        Rect cleanedDetection)
    {
        if (originalTarget.Width <= 0 ||
            originalTarget.Height <= 0 ||
            cleanedDetection.Width <= 0 ||
            cleanedDetection.Height <= 0)
        {
            return false;
        }

        int margin =
            Math.Max(
                4,
                (int)Math.Round(
                    Math.Min(
                        originalTarget.Width,
                        originalTarget.Height) *
                    0.06));

        var expanded =
            new Rect(
                originalTarget.X - margin,
                originalTarget.Y - margin,
                originalTarget.Width + margin * 2,
                originalTarget.Height + margin * 2);

        double intersection =
            IntersectionArea(
                expanded,
                cleanedDetection);

        if (intersection <= 0)
            return false;

        double detectionArea =
            Math.Max(
                1,
                cleanedDetection.Width *
                (double)cleanedDetection.Height);

        double targetArea =
            Math.Max(
                1,
                expanded.Width *
                (double)expanded.Height);

        double detectionCoverage =
            intersection /
            detectionArea;

        double targetCoverage =
            intersection /
            targetArea;

        double centerX =
            cleanedDetection.X +
            cleanedDetection.Width / 2.0;

        double centerY =
            cleanedDetection.Y +
            cleanedDetection.Height / 2.0;

        bool centerInside =
            centerX >= expanded.Left &&
            centerX <= expanded.Right &&
            centerY >= expanded.Top &&
            centerY <= expanded.Bottom;

        return detectionCoverage >= 0.45 ||
               targetCoverage >= 0.18 ||
               (centerInside &&
                detectionCoverage >= 0.20);
    }

    static IEnumerable<V2CleanedStateCheckpoint>
        BuildCheckpoints(
            IReadOnlyList<V2TextTarget> targets,
            IReadOnlyDictionary<string, V2CleanedTargetVerification> checks)
    {
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

            bool empty =
                ids.Length > 0 &&
                ids.All(id =>
                    checks.TryGetValue(
                        id,
                        out var check) &&
                    check.EmptyVerified);

            bool legacyClean =
                ids.Length > 0 &&
                ids.All(id =>
                    checks.TryGetValue(
                        id,
                        out var check) &&
                    check.LegacyReviewerClean);

            Rect bounds =
                group.Select(x =>
                        x.BubbleBounds)
                    .FirstOrDefault(x =>
                        x.HasValue) ??
                UnionBounds(
                    group.Select(x =>
                        x.TextBounds));

            yield return new V2CleanedStateCheckpoint(
                group.Key,
                "Bubble",
                bounds,
                ids,
                empty,
                empty
                    ? "EMPTY_OK"
                    : "TEXT_REDETECTED",
                legacyClean);
        }

        foreach (var target in targets.Where(x =>
                     x.Kind ==
                     PageRegionKind.TextFree))
        {
            bool empty =
                checks.TryGetValue(
                    target.TextRegionId,
                    out var check) &&
                check.EmptyVerified;

            bool legacyClean =
                check?.LegacyReviewerClean ??
                false;

            yield return new V2CleanedStateCheckpoint(
                target.TextRegionId,
                "TextFree",
                target.TextBounds,
                [target.TextRegionId],
                empty,
                empty
                    ? "EMPTY_OK"
                    : "TEXT_REDETECTED",
                legacyClean);
        }

        foreach (var target in targets.Where(x =>
                     x.Kind ==
                         PageRegionKind.TextBubble &&
                     x.BubbleRegionId is null))
        {
            bool empty =
                checks.TryGetValue(
                    target.TextRegionId,
                    out var check) &&
                check.EmptyVerified;

            bool legacyClean =
                check?.LegacyReviewerClean ??
                false;

            yield return new V2CleanedStateCheckpoint(
                target.TextRegionId,
                "UnparentedTextBubble",
                target.TextBounds,
                [target.TextRegionId],
                empty,
                empty
                    ? "EMPTY_OK"
                    : "TEXT_REDETECTED",
                legacyClean);
        }
    }

    static void SaveDebug(
        string cleanedPath,
        V2DetectionSnapshot snapshot,
        IReadOnlyList<V2CleanedStateCheckpoint> checkpoints,
        IReadOnlyList<PageRegion> residualTextRegions,
        string path)
    {
        using var image =
            Cv2.ImRead(
                cleanedPath,
                ImreadModes.Color);

        if (image.Empty())
            return;

        var targetBubbleIds =
            checkpoints
                .Where(x =>
                    x.Kind == "Bubble")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        foreach (var region in snapshot.RawRegions.Where(x =>
                     x.Kind ==
                         PageRegionKind.Bubble &&
                     !targetBubbleIds.Contains(
                         x.RegionId)))
        {
            Cv2.Rectangle(
                image,
                region.Bounds,
                new Scalar(
                    105,
                    105,
                    105),
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
                image,
                checkpoint.Bounds,
                color,
                3);

            string label =
                $"{checkpoint.CheckpointId} {checkpoint.Status}";

            Cv2.PutText(
                image,
                label,
                new Point(
                    Math.Max(
                        2,
                        checkpoint.Bounds.X),
                    Math.Max(
                        16,
                        checkpoint.Bounds.Y - 5)),
                HersheyFonts.HersheySimplex,
                0.45,
                color,
                1,
                LineTypes.AntiAlias);
        }

        foreach (var residual in residualTextRegions)
        {
            Cv2.Rectangle(
                image,
                residual.Bounds,
                new Scalar(
                    255,
                    0,
                    255),
                2);
        }

        int bubbleTargets =
            checkpoints.Count(x =>
                x.Kind == "Bubble");

        int bubbleEmpty =
            checkpoints.Count(x =>
                x.Kind == "Bubble" &&
                x.EmptyVerified);

        int textFreeTargets =
            checkpoints.Count(x =>
                x.Kind == "TextFree");

        int textFreeEmpty =
            checkpoints.Count(x =>
                x.Kind == "TextFree" &&
                x.EmptyVerified);

        string summary =
            $"Bubble {bubbleEmpty}/{bubbleTargets} EMPTY | " +
            $"TextFree {textFreeEmpty}/{textFreeTargets} EMPTY | " +
            $"Cleaned text det {residualTextRegions.Count}";

        Cv2.Rectangle(
            image,
            new Rect(
                0,
                0,
                Math.Min(
                    image.Cols,
                    Math.Max(
                        360,
                        summary.Length * 11)),
                Math.Min(
                    image.Rows,
                    34)),
            new Scalar(
                20,
                20,
                20),
            thickness: -1);

        Cv2.PutText(
            image,
            summary,
            new Point(
                8,
                23),
            HersheyFonts.HersheySimplex,
            0.55,
            Scalar.White,
            1,
            LineTypes.AntiAlias);

        Cv2.ImWrite(
            path,
            image,
            [
                new ImageEncodingParam(
                    ImwriteFlags.WebPQuality,
                    LosslessWebpQuality)
            ]);
    }

    static void WriteAudit(
        string sourcePath,
        bool verifierAvailable,
        string verifierMode,
        V2DetectionSnapshot snapshot,
        IReadOnlyList<V2TextTarget> targets,
        IReadOnlyList<V2CleanedTargetVerification> targetChecks,
        IReadOnlyList<V2CleanedStateCheckpoint> checkpoints,
        IReadOnlyList<PageRegion> residualTextRegions,
        string path)
    {
        var eraseTargetBubbleIds =
            checkpoints
                .Where(x =>
                    x.Kind == "Bubble")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var emptyBubbleIds =
            checkpoints
                .Where(x =>
                    x.Kind == "Bubble" &&
                    x.EmptyVerified)
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var eraseTargetTextFreeIds =
            checkpoints
                .Where(x =>
                    x.Kind == "TextFree")
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var emptyTextFreeIds =
            checkpoints
                .Where(x =>
                    x.Kind == "TextFree" &&
                    x.EmptyVerified)
                .Select(x =>
                    x.CheckpointId)
                .ToHashSet(
                    StringComparer.Ordinal);

        var document =
            new
            {
                Schema =
                    "pipeline-v2-cleaned-state-v1",
                SourceFile =
                    Path.GetFileName(
                        sourcePath),
                VerifierAvailable =
                    verifierAvailable,
                VerifierMode =
                    verifierMode,
                OriginalDetectedBubbleCount =
                    snapshot.RawRegions.Count(x =>
                        x.Kind ==
                        PageRegionKind.Bubble),
                SelectedTargetCount =
                    targets.Count,
                ResidualTextDetectionCount =
                    residualTextRegions.Count,
                EraseTargetBubbleCount =
                    eraseTargetBubbleIds.Count,
                EmptyVerifiedBubbleCount =
                    emptyBubbleIds.Count,
                BubbleCheckpointPass =
                    ErasePipelineV2.CheckpointIdsMatch(
                        eraseTargetBubbleIds,
                        emptyBubbleIds),
                EraseTargetBubbleIds =
                    eraseTargetBubbleIds
                        .OrderBy(x => x)
                        .ToArray(),
                EmptyVerifiedBubbleIds =
                    emptyBubbleIds
                        .OrderBy(x => x)
                        .ToArray(),
                CheckpointMissingBubbleIds =
                    eraseTargetBubbleIds
                        .Except(
                            emptyBubbleIds,
                            StringComparer.Ordinal)
                        .OrderBy(x => x)
                        .ToArray(),
                EraseTargetTextFreeCount =
                    eraseTargetTextFreeIds.Count,
                EmptyVerifiedTextFreeCount =
                    emptyTextFreeIds.Count,
                TextFreeCheckpointPass =
                    ErasePipelineV2.CheckpointIdsMatch(
                        eraseTargetTextFreeIds,
                        emptyTextFreeIds),
                OverallPass =
                    ErasePipelineV2.CheckpointIdsMatch(
                        eraseTargetBubbleIds,
                        emptyBubbleIds) &&
                    ErasePipelineV2.CheckpointIdsMatch(
                        eraseTargetTextFreeIds,
                        emptyTextFreeIds),
                LegacyReviewerDisagreements =
                    targetChecks.Count(x =>
                        x.EmptyVerified !=
                        x.LegacyReviewerClean),
                Targets =
                    targetChecks,
                Checkpoints =
                    checkpoints,
                ResidualDetections =
                    residualTextRegions.Select(x =>
                        new
                        {
                            x.RegionId,
                            x.Kind,
                            Bounds =
                                new
                                {
                                    x.Bounds.X,
                                    x.Bounds.Y,
                                    x.Bounds.Width,
                                    x.Bounds.Height
                                },
                            x.Score
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

    static double IntersectionArea(
        Rect a,
        Rect b)
    {
        int left =
            Math.Max(
                a.Left,
                b.Left);

        int top =
            Math.Max(
                a.Top,
                b.Top);

        int right =
            Math.Min(
                a.Right,
                b.Right);

        int bottom =
            Math.Min(
                a.Bottom,
                b.Bottom);

        return Math.Max(
                   0,
                   right - left) *
               (double)Math.Max(
                   0,
                   bottom - top);
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
}
