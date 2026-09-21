using System.Globalization;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.PipelineV2.Erase;
using OpenCvSharp;
using CvRect = OpenCvSharp.Rect;
using WpfRect = System.Windows.Rect;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Transitional render pipeline.
///
/// Phase 1 goal:
/// - OCR observations, container ownership, erase planning and layout planning are explicit.
/// - layout must succeed before any pixels are erased.
/// - every approved unit owns one canonical container and is rendered exactly once.
/// - final erase pixels are always clipped to an explicit allowed mask.
/// - debug output is separated by pipeline stage.
/// </summary>
public sealed class RenderPipelineService
{
    const double InpaintRadius = 2.5;
    const int FinalWebpLosslessQuality = 101;
    const int DebugWebpQuality = 82;

    readonly CleanedStateVerifier cleanedStateVerifier =
        new();

    sealed record ContainerSelection(
        VisionTranslation Region,
        BalloonLayout Layout,
        string ContainerId);

    sealed record CanonicalLine(
        string LineId,
        OcrLine Line);

    public async Task RenderAsync(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        string outputPath,
        IProgress<string>? progress = null,
        CancellationToken token = default,
        V2DetectionSnapshot? v2Snapshot = null,
        V2EraseSelection? v2Selection = null)
    {
        bool useV2 =
            v2Snapshot is not null &&
            v2Selection is not null;

        var allRequested =
            regions
                .Where(x =>
                    x.Render &&
                    !string.IsNullOrWhiteSpace(
                        x.Translation))
                .ToList();

        var requested =
            useV2
                ? allRequested
                    .Where(x =>
                        v2Selection!.Bindings.ContainsKey(
                            x.Id))
                    .ToList()
                : allRequested;

        List<VisionTranslation> legacySuppressed = [];

        var candidates =
            useV2
                ? requested
                : SuppressDuplicateRegions(
                    requested,
                    out legacySuppressed);

        var debugDir =
            Path.Combine(
                Path.GetDirectoryName(outputPath) ??
                    AppContext.BaseDirectory,
                "debug");

        Directory.CreateDirectory(
            debugDir);

        string baseName =
            Path.GetFileNameWithoutExtension(
                outputPath);

        string sourceBaseName =
            Path.GetFileNameWithoutExtension(
                sourcePath);

        string ocrDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.01_planned_lines.webp");

        string containerDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.02_container_assignment.webp");

        string unitDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.03_translation_units.webp");

        string eraseDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.04_erase_mask.webp");

        string layoutDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.05_layout_box.webp");

        string finalDebug =
            Path.Combine(
                debugDir,
                $"{baseName}.06_final_output.webp");

        string planJson =
            Path.Combine(
                debugDir,
                $"{baseName}.render-plan.json");

        string v2CommittedCleaned =
            Path.Combine(
                debugDir,
                $"{sourceBaseName}.v2_05_committed_cleaned.webp");

        string v2CommitAudit =
            Path.Combine(
                debugDir,
                $"{sourceBaseName}.v2_commit_audit.json");

        string v2TargetCoverageAudit =
            Path.Combine(
                debugDir,
                $"{sourceBaseName}.v2_target_coverage_audit.json");

        if (candidates.Count == 0)
        {
            SaveSourceAsOutput(
                sourcePath,
                outputPath);

            SaveDebugCopy(
                outputPath,
                finalDebug);

            if (useV2)
            {
                WriteV2CommitAudit(
                    sourcePath,
                    allRequested,
                    v2Selection!,
                    [],
                    [],
                    [],
                    [],
                    [],
                    v2CommitAudit);

                WriteV2TargetCoverageAudit(
                    sourcePath,
                    regions,
                    v2Snapshot!,
                    v2Selection!,
                    [],
                    [],
                    v2TargetCoverageAudit);
            }

            progress?.Report(
                useV2
                    ? "Pipeline V2 · 연결 가능한 번역 Unit이 없어 원문을 유지했습니다."
                    : "조판할 번역 영역이 없어 원문을 유지했습니다.");

            return;
        }

        if (legacySuppressed.Count > 0)
        {
            progress?.Report(
                $"기존 중복 방어 {legacySuppressed.Count}개 제외 · " +
                string.Join(
                    ", ",
                    legacySuppressed.Select(x =>
                        $"id={x.Id}")));
        }

        progress?.Report(
            useV2
                ? $"1/5 V2 parent Bubble 조판영역 고정 · {candidates.Count}개 번역 블록"
                : $"1/5 컨테이너 분석 · {candidates.Count}개 번역 블록");

        var layouts =
            useV2
                ? CreateV2BubbleLayouts(
                    candidates,
                    v2Selection!.Bindings,
                    progress)
                : await Task.Run(
                    () => AnalyzeContainers(
                        sourcePath,
                        candidates,
                        progress,
                        token),
                    token);

        token.ThrowIfCancellationRequested();

        progress?.Report(
            "2/5 ContainerId / UnitId / LineId 확정 + 조판 사전 검증");

        var plans =
            BuildRenderPlans(
                candidates,
                layouts,
                out var containerSuppressed,
                progress,
                requireLegacyErase:
                    !useV2);

        if (containerSuppressed.Count > 0)
        {
            progress?.Report(
                $"동일 컨테이너 중복 {containerSuppressed.Count}개 제외 · " +
                string.Join(
                    ", ",
                    containerSuppressed.Select(x =>
                        $"id={x.Id}")));
        }

        int preApproved =
            plans.Count(x =>
                x.Approved);

        int preRejected =
            plans.Count -
            preApproved;

        progress?.Report(
            $"렌더 계획 · 승인 {preApproved} · 원문 유지 {preRejected}");

        await Task.Run(
            () => SavePlanningDebug(
                sourcePath,
                plans,
                ocrDebug,
                containerDebug,
                unitDebug,
                layoutDebug,
                token),
            token);

        if (useV2)
        {
            try
            {
                if (File.Exists(
                        eraseDebug))
                {
                    File.Delete(
                        eraseDebug);
                }
            }
            catch
            {
                // stale legacy debug must not fail V2 rendering.
            }

            var layoutApprovedPlans =
                plans
                    .Where(x =>
                        x.Approved)
                    .ToList();

            var eraseTargetIds =
                layoutApprovedPlans
                    .Where(x =>
                        v2Selection!.Bindings.ContainsKey(
                            x.Region.Id))
                    .SelectMany(x =>
                        v2Selection!.Bindings[
                            x.Region.Id]
                            .TextRegionIds)
                    .ToHashSet(
                        StringComparer.Ordinal);

            if (layoutApprovedPlans.Count == 0 ||
                eraseTargetIds.Count == 0)
            {
                SaveSourceAsOutput(
                    sourcePath,
                    outputPath);

                SaveDebugCopy(
                    outputPath,
                    finalDebug);

                WritePlanJson(
                    sourcePath,
                    plans,
                    planJson);

                WriteV2CommitAudit(
                    sourcePath,
                    allRequested,
                    v2Selection!,
                    plans,
                    containerSuppressed,
                    [],
                    [],
                    [],
                    v2CommitAudit);

                WriteV2TargetCoverageAudit(
                    sourcePath,
                    regions,
                    v2Snapshot!,
                    v2Selection!,
                    plans,
                    [],
                    v2TargetCoverageAudit);

                progress?.Report(
                    "Pipeline V2 · 조판 가능한 Unit이 없어 원문을 유지했습니다.");

                return;
            }

            progress?.Report(
                $"3/5 조판 가능 Unit {layoutApprovedPlans.Count}개만 V2 erase 검수");

            string outputDirectory =
                Path.GetDirectoryName(
                    outputPath) ??
                AppContext.BaseDirectory;

            var v2Erase =
                await Task.Run(
                    () => new ErasePipelineV2().Run(
                        sourcePath,
                        outputDirectory,
                        v2Snapshot!,
                        eraseTargetIds,
                        token),
                    token);

            progress?.Report(
                "3/5 erase 완료 · cleaned 이미지에서 RT-DETR 텍스트 재검출");

            var cleanedState =
                await Task.Run(
                    () => cleanedStateVerifier.Verify(
                        sourcePath,
                        v2Erase.CleanedDebugPath,
                        outputDirectory,
                        v2Snapshot!,
                        eraseTargetIds,
                        v2Erase.TargetAudits,
                        token),
                    token);

            var cleanedByTarget =
                cleanedState.Targets
                    .ToDictionary(
                        x => x.TextRegionId,
                        StringComparer.Ordinal);

            var backgroundQualityFailedIds =
                v2Erase.BackgroundQualityFailedTextRegionIds
                    .ToHashSet(
                        StringComparer.Ordinal);

            int legacyDisagreements =
                cleanedState.Targets.Count(x =>
                    x.EmptyVerified !=
                    x.LegacyReviewerClean);

            progress?.Report(
                $"Cleaned checkpoint · mode={cleanedState.VerifierMode} · " +
                $"target={cleanedState.Targets.Count} · " +
                $"empty={cleanedState.Targets.Count(x => x.EmptyVerified)} · " +
                $"background quality fail={backgroundQualityFailedIds.Count} · " +
                $"legacy disagreement={legacyDisagreements}");

            bool CleanedStateEmptyFor(
                RenderUnitPlan plan)
            {
                if (!v2Selection!.Bindings.TryGetValue(
                        plan.Region.Id,
                        out var binding) ||
                    binding.TextRegionIds.Count == 0)
                {
                    return false;
                }

                return binding.TextRegionIds.All(id =>
                    cleanedByTarget.TryGetValue(
                        id,
                        out var check) &&
                    check.EmptyVerified &&
                    !backgroundQualityFailedIds.Contains(
                        id));
            }

            var finalPlans =
                layoutApprovedPlans
                    .Where(
                        CleanedStateEmptyFor)
                    .ToList();

            var committedTargetIds =
                finalPlans
                    .SelectMany(plan =>
                        v2Selection!.Bindings[
                            plan.Region.Id]
                            .TextRegionIds)
                    .ToHashSet(
                        StringComparer.Ordinal);

            BuildCommittedV2Cleaned(
                sourcePath,
                v2Erase.CleanedDebugPath,
                v2Snapshot!,
                committedTargetIds,
                v2CommittedCleaned);

            int eraseRejected =
                layoutApprovedPlans.Count -
                finalPlans.Count;

            progress?.Report(
                $"4/5 cleaned-state commit · empty 확인 {finalPlans.Count} · " +
                $"텍스트 재검출/검수실패 {eraseRejected} · " +
                $"지운 Unit={finalPlans.Count} / 채울 Unit={finalPlans.Count}");

            if (finalPlans.Count == 0)
            {
                SaveSourceAsOutput(
                    sourcePath,
                    outputPath);
            }
            else
            {
                TypesetAndSave(
                    v2CommittedCleaned,
                    finalPlans,
                    outputPath,
                    token);
            }

            SaveDebugCopy(
                outputPath,
                finalDebug);

            var committedUnitIds =
                finalPlans
                    .Select(x => x.UnitId)
                    .ToHashSet(
                        StringComparer.Ordinal);

            var finalizedPlans =
                plans
                    .Select(plan =>
                        plan.Approved &&
                        !committedUnitIds.Contains(
                            plan.UnitId)
                            ? plan with
                            {
                                Approved = false,
                                Reason = "v2_commit_preserved_original"
                            }
                            : plan)
                    .ToList();

            WritePlanJson(
                sourcePath,
                finalizedPlans,
                planJson);

            WriteV2CommitAudit(
                sourcePath,
                allRequested,
                v2Selection!,
                plans,
                containerSuppressed,
                v2Erase.TargetAudits,
                cleanedState.Targets,
                finalPlans,
                v2CommitAudit);

            WriteV2TargetCoverageAudit(
                sourcePath,
                regions,
                v2Snapshot!,
                v2Selection!,
                plans,
                finalPlans,
                v2TargetCoverageAudit);

            progress?.Report(
                $"5/5 완료 · 원본 Unit 보존/교체 1:1 · " +
                $"erase={finalPlans.Count} typeset={finalPlans.Count} · " +
                $"{Path.GetFileName(outputPath)}");

            return;
        }

        string cleanedPath =
            Path.Combine(
                Path.GetTempPath(),
                $"lmt_plan_inpaint_{Guid.NewGuid():N}.png");

        try
        {
            progress?.Report(
                "3/5 승인된 Unit의 삭제 마스크 생성 + 허용영역 최종 제한");

            var erasedUnitIds =
                await Task.Run(
                    () => InpaintApprovedUnits(
                        sourcePath,
                        cleanedPath,
                        eraseDebug,
                        plans,
                        progress,
                        token),
                    token);

            var finalPlans =
                plans
                    .Where(x =>
                        x.Approved &&
                        erasedUnitIds.Contains(
                            x.UnitId))
                    .ToList();

            int eraseRejected =
                preApproved -
                finalPlans.Count;

            if (eraseRejected > 0)
            {
                progress?.Report(
                    $"삭제 마스크 불충분 {eraseRejected}개는 원문 유지");
            }

            if (finalPlans.Count == 0)
            {
                SaveSourceAsOutput(
                    sourcePath,
                    outputPath);

                SaveDebugCopy(
                    outputPath,
                    finalDebug);

                WritePlanJson(
                    sourcePath,
                    plans,
                    planJson);

                progress?.Report(
                    "안전하게 조판할 Unit이 없어 원문을 유지했습니다.");

                return;
            }

            token.ThrowIfCancellationRequested();

            progress?.Report(
                $"4/5 승인 Unit {finalPlans.Count}개를 계획대로 1회 조판");

            TypesetAndSave(
                cleanedPath,
                finalPlans,
                outputPath,
                token);

            SaveDebugCopy(
                outputPath,
                finalDebug);

            WritePlanJson(
                sourcePath,
                plans,
                planJson);

            progress?.Report(
                $"5/5 완료 · RenderPlan {finalPlans.Count}개 · {Path.GetFileName(outputPath)}");
        }
        finally
        {
            try
            {
                if (File.Exists(
                        cleanedPath))
                {
                    File.Delete(
                        cleanedPath);
                }
            }
            catch
            {
                // temp cleanup failure does not invalidate output.
            }
        }
    }

    static Dictionary<int, BalloonLayout> CreateV2BubbleLayouts(
        IReadOnlyList<VisionTranslation> regions,
        IReadOnlyDictionary<int, V2RegionBinding> bindings,
        IProgress<string>? progress)
    {
        var result =
            new Dictionary<int, BalloonLayout>(
                regions.Count);

        foreach (var region in regions)
        {
            if (!bindings.TryGetValue(
                    region.Id,
                    out var binding))
            {
                continue;
            }

            var bounds =
                binding.LayoutBounds;

            int insetX =
                Math.Clamp(
                    (int)Math.Round(
                        bounds.Width *
                        0.08),
                    4,
                    24);

            int insetY =
                Math.Clamp(
                    (int)Math.Round(
                        bounds.Height *
                        0.10),
                    4,
                    24);

            int innerWidth =
                Math.Max(
                    12,
                    bounds.Width -
                    insetX * 2);

            int innerHeight =
                Math.Max(
                    12,
                    bounds.Height -
                    insetY * 2);

            var inner =
                new CvRect(
                    bounds.X +
                        insetX,
                    bounds.Y +
                        insetY,
                    innerWidth,
                    innerHeight);

            int maskWidth =
                Math.Max(
                    1,
                    bounds.Width);

            int maskHeight =
                Math.Max(
                    1,
                    bounds.Height);

            var safeMask =
                Enumerable.Repeat(
                        (byte)255,
                        maskWidth *
                        maskHeight)
                    .ToArray();

            var layout =
                new BalloonLayout(
                    bounds,
                    inner,
                    true,
                    false,
                    safeMask,
                    maskWidth,
                    maskHeight,
                    region.Type,
                    binding.LayoutMode switch
                    {
                        "rtdetr_parent_bubble" =>
                            "v2:rtdetr_parent_bubble",
                        "rtdetr_textfree" =>
                            "v2:rtdetr_textfree",
                        _ =>
                            "v2:textbubble_layout_fallback"
                    },
                    true,
                    binding.LayoutMode switch
                    {
                        "rtdetr_parent_bubble" =>
                            "v2_parent_bubble_for_layout",
                        "rtdetr_textfree" =>
                            "v2_textfree_local_layout",
                        _ =>
                            "v2_no_parent_bubble"
                    },
                    1.0,
                    1.0,
                    1.0,
                    0,
                    inner.Width *
                    (double)inner.Height /
                    Math.Max(
                        1.0,
                        bounds.Width *
                        (double)bounds.Height),
                    1.0,
                    0);

            result[
                region.Id] =
                layout;

            progress?.Report(
                $"[v2-box] id={region.Id} " +
                $"erase_text={binding.TextBounds.X},{binding.TextBounds.Y}," +
                $"{binding.TextBounds.Width}x{binding.TextBounds.Height} " +
                $"layout={bounds.X},{bounds.Y},{bounds.Width}x{bounds.Height} " +
                $"mode={binding.LayoutMode}");
        }

        return result;
    }

    static void BuildCommittedV2Cleaned(
        string sourcePath,
        string v2CleanedPath,
        V2DetectionSnapshot snapshot,
        IReadOnlySet<string> committedTargetIds,
        string outputPath)
    {
        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        using var cleaned =
            Cv2.ImRead(
                v2CleanedPath,
                ImreadModes.Color);

        if (source.Empty() ||
            cleaned.Empty())
        {
            throw new InvalidOperationException(
                "Pipeline V2 commit 이미지를 열 수 없습니다.");
        }

        using var committed =
            source.Clone();

        foreach (var target in snapshot.TextTargets.Where(x =>
                     committedTargetIds.Contains(
                         x.TextRegionId)))
        {
            var bounds =
                ClampRect(
                    target.TextBounds.X,
                    target.TextBounds.Y,
                    target.TextBounds.Width,
                    target.TextBounds.Height,
                    source.Cols,
                    source.Rows);

            if (bounds.Width <= 0 ||
                bounds.Height <= 0)
            {
                continue;
            }

            using var srcRoi =
                new Mat(
                    cleaned,
                    bounds);

            using var dstRoi =
                new Mat(
                    committed,
                    bounds);

            srcRoi.CopyTo(
                dstRoi);
        }

        if (!SaveLosslessWebp(
                outputPath,
                committed))
        {
            throw new InvalidOperationException(
                "Pipeline V2 committed cleaned 이미지를 저장하지 못했습니다.");
        }
    }

    static void WriteV2CommitAudit(
        string sourcePath,
        IReadOnlyList<VisionTranslation> requested,
        V2EraseSelection selection,
        IReadOnlyList<RenderUnitPlan> plans,
        IReadOnlyList<VisionTranslation> containerSuppressed,
        IReadOnlyList<V2EraseTargetAudit> targetAudits,
        IReadOnlyList<V2CleanedTargetVerification> cleanedVerifications,
        IReadOnlyList<RenderUnitPlan> committedPlans,
        string path)
    {
        try
        {
            var planByRegion =
                plans.ToDictionary(
                    x => x.Region.Id);

            var suppressedIds =
                containerSuppressed
                    .Select(x => x.Id)
                    .ToHashSet();

            var auditByTarget =
                targetAudits
                    .ToDictionary(
                        x => x.TextRegionId,
                        StringComparer.Ordinal);

            var cleanedByTarget =
                cleanedVerifications
                    .ToDictionary(
                        x => x.TextRegionId,
                        StringComparer.Ordinal);

            var committedIds =
                committedPlans
                    .Select(x => x.Region.Id)
                    .ToHashSet();

            var units =
                requested
                    .Select(region =>
                    {
                        bool bound =
                            selection.Bindings.TryGetValue(
                                region.Id,
                                out var binding);

                        bool planned =
                            planByRegion.TryGetValue(
                                region.Id,
                                out var plan);

                        bool layoutReady =
                            planned &&
                            plan!.Approved;

                        bool legacyEraseClean =
                            bound &&
                            binding!.TextRegionIds.Count > 0 &&
                            binding.TextRegionIds.All(id =>
                                auditByTarget.TryGetValue(
                                    id,
                                    out var audit) &&
                                ErasePipelineV2.IsAuditClean(
                                    audit));

                        bool cleanedStateEmpty =
                            bound &&
                            binding!.TextRegionIds.Count > 0 &&
                            binding.TextRegionIds.All(id =>
                                cleanedByTarget.TryGetValue(
                                    id,
                                    out var cleaned) &&
                                cleaned.EmptyVerified);

                        bool cleanedVerifierAvailable =
                            bound &&
                            binding!.TextRegionIds.Count > 0 &&
                            binding.TextRegionIds.All(id =>
                                cleanedByTarget.ContainsKey(
                                    id));

                        bool committed =
                            committedIds.Contains(
                                region.Id);

                        string? policyReason =
                            null;

                        if (!bound)
                        {
                            selection.PreservationReasons.TryGetValue(
                                region.Id,
                                out policyReason);
                        }

                        bool auditNoise =
                            !bound &&
                            IsLikelyAuditOcrNoise(
                                region);

                        string status;

                        if (committed)
                        {
                            status =
                                "translated";
                        }
                        else if (!string.IsNullOrWhiteSpace(
                                     policyReason))
                        {
                            status =
                                $"original_preserved:{policyReason}";
                        }
                        else if (auditNoise)
                        {
                            status =
                                "original_preserved:ocr_noise";
                        }
                        else if (!bound)
                        {
                            status =
                                "original_preserved:unbound";
                        }
                        else if (suppressedIds.Contains(
                                     region.Id))
                        {
                            status =
                                "original_preserved:container_duplicate";
                        }
                        else if (!planned)
                        {
                            status =
                                "original_preserved:not_planned";
                        }
                        else if (!layoutReady)
                        {
                            status =
                                $"original_preserved:{plan!.Reason}";
                        }
                        else if (cleanedVerifierAvailable &&
                                 !cleanedStateEmpty)
                        {
                            status =
                                "original_preserved:cleaned_text_redetected";
                        }
                        else if (!cleanedVerifierAvailable &&
                                 !legacyEraseClean)
                        {
                            status =
                                "original_preserved:erase_review_failed";
                        }
                        else
                        {
                            status =
                                "original_preserved:not_committed";
                        }

                        return new
                        {
                            RegionId =
                                region.Id,
                            SourceText =
                                region.Source.Text,
                            Translation =
                                region.Translation,
                            Bound =
                                bound,
                            TextRegionIds =
                                bound
                                    ? binding!.TextRegionIds
                                    : [],
                            BubbleRegionId =
                                bound
                                    ? binding!.BubbleRegionId
                                    : null,
                            LayoutMode =
                                bound
                                    ? binding!.LayoutMode
                                    : null,
                            LayoutBounds =
                                bound
                                    ? new
                                    {
                                        binding!.LayoutBounds.X,
                                        binding.LayoutBounds.Y,
                                        binding.LayoutBounds.Width,
                                        binding.LayoutBounds.Height
                                    }
                                    : null,
                            Planned =
                                planned,
                            LayoutApproved =
                                layoutReady,
                            LegacyEraseReviewClean =
                                legacyEraseClean,
                            CleanedStateVerifierAvailable =
                                cleanedVerifierAvailable,
                            CleanedStateEmpty =
                                cleanedStateEmpty,
                            ReviewerDisagreement =
                                cleanedVerifierAvailable &&
                                legacyEraseClean !=
                                cleanedStateEmpty,
                            EraseClean =
                                cleanedVerifierAvailable
                                    ? cleanedStateEmpty
                                    : legacyEraseClean,
                            Committed =
                                committed,
                            Status =
                                status
                        };
                    })
                    .ToArray();

            int committedCount =
                units.Count(x =>
                    x.Committed);

            int preservedCount =
                units.Length -
                committedCount;

            var document =
                new
                {
                    Schema =
                        "pipeline-v2-cleaned-state-commit-v2",
                    SourceFile =
                        Path.GetFileName(
                            sourcePath),
                    RequestedUnits =
                        units.Length,
                    TranslatedUnits =
                        committedCount,
                    PreservedOriginalUnits =
                        preservedCount,
                    EraseCommittedUnits =
                        committedCount,
                    TypesetCommittedUnits =
                        committedCount,
                    MissingUnits =
                        0,
                    DuplicateUnits =
                        0,
                    EraseTypesetCountMatch =
                        true,
                    SourceToFinalCountMatch =
                        units.Length ==
                        committedCount +
                        preservedCount,
                    Units =
                        units
                };

            File.WriteAllText(
                path,
                JsonSerializer.Serialize(
                    document,
                    new JsonSerializerOptions
                    {
                        WriteIndented =
                            true
                    }));
        }
        catch
        {
            // diagnostic failure must not invalidate a finished image.
        }
    }

    static void WriteV2TargetCoverageAudit(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        V2DetectionSnapshot snapshot,
        V2EraseSelection selection,
        IReadOnlyList<RenderUnitPlan> plans,
        IReadOnlyList<RenderUnitPlan> finalPlans,
        string path)
    {
        try
        {
            var committedRegionIds =
                finalPlans
                    .Select(x =>
                        x.Region.Id)
                    .ToHashSet();

            var plansByRegion =
                plans
                    .GroupBy(x =>
                        x.Region.Id)
                    .ToDictionary(
                        x => x.Key,
                        x => x
                            .OrderByDescending(y =>
                                y.Approved)
                            .ThenBy(y =>
                                y.UnitId)
                            .First());

            var targets =
                snapshot.TextTargets
                    .Select(target =>
                    {
                        var boundRegionIds =
                            selection.Bindings
                                .Where(x =>
                                    x.Value.TextRegionIds.Contains(
                                        target.TextRegionId,
                                        StringComparer.Ordinal))
                                .Select(x =>
                                    x.Key)
                                .Distinct()
                                .OrderBy(x =>
                                    x)
                                .ToArray();

                        int? committedRegionId =
                            boundRegionIds
                                .FirstOrDefault(id =>
                                    committedRegionIds.Contains(
                                        id));

                        bool committed =
                            committedRegionId.HasValue &&
                            committedRegionId.Value != 0;

                        string status;
                        int? bestRegionId =
                            null;
                        string? planReason =
                            null;

                        if (committed)
                        {
                            status =
                                "translated_committed";

                            bestRegionId =
                                committedRegionId;
                        }
                        else if (selection.PreservationTargetReasons.TryGetValue(
                                     target.TextRegionId,
                                     out var policyReason))
                        {
                            status =
                                $"preserved_policy:{policyReason}";
                        }
                        else if (boundRegionIds.Length > 0)
                        {
                            var boundPlans =
                                boundRegionIds
                                    .Where(id =>
                                        plansByRegion.ContainsKey(
                                            id))
                                    .Select(id =>
                                        plansByRegion[id])
                                    .OrderByDescending(x =>
                                        x.Approved)
                                    .ThenByDescending(x =>
                                        V2CandidateKeepScore(
                                            x.Region))
                                    .ToList();

                            if (boundPlans.Count == 0)
                            {
                                status =
                                    "unresolved_bound_without_plan";

                                bestRegionId =
                                    boundRegionIds[0];
                            }
                            else
                            {
                                var bestPlan =
                                    boundPlans[0];

                                bestRegionId =
                                    bestPlan.Region.Id;

                                planReason =
                                    bestPlan.Reason;

                                status =
                                    bestPlan.Approved
                                        ? "preserved_cleaned_state_or_background_quality"
                                        : $"preserved_plan:{bestPlan.Reason}";
                            }
                        }
                        else
                        {
                            var bestRegion =
                                regions
                                    .Select(region => new
                                    {
                                        Region = region,
                                        Score =
                                            TargetAffinityScore(
                                                region,
                                                target)
                                    })
                                    .Where(x =>
                                        x.Score > 0)
                                    .OrderByDescending(x =>
                                        x.Score)
                                    .ThenBy(x =>
                                        x.Region.Id)
                                    .FirstOrDefault();

                            if (bestRegion is null)
                            {
                                status =
                                    "unresolved_detected_no_translation_unit";
                            }
                            else
                            {
                                bestRegionId =
                                    bestRegion.Region.Id;

                                if (!bestRegion.Region.Render ||
                                    string.IsNullOrWhiteSpace(
                                        bestRegion.Region.Translation))
                                {
                                    status =
                                        $"preserved_render_false:{bestRegion.Region.Type}";
                                }
                                else
                                {
                                    status =
                                        "unresolved_detected_unbound_translation";
                                }
                            }
                        }

                        bool unresolved =
                            status.StartsWith(
                                "unresolved_",
                                StringComparison.Ordinal);

                        return new
                        {
                            TextRegionId =
                                target.TextRegionId,
                            Kind =
                                target.Kind.ToString(),
                            target.TextScore,
                            BubbleRegionId =
                                target.BubbleRegionId,
                            TextBounds =
                                new
                                {
                                    target.TextBounds.X,
                                    target.TextBounds.Y,
                                    target.TextBounds.Width,
                                    target.TextBounds.Height
                                },
                            BoundTranslationRegionIds =
                                boundRegionIds,
                            BestTranslationRegionId =
                                bestRegionId,
                            Status =
                                status,
                            PlanReason =
                                planReason,
                            Unresolved =
                                unresolved
                        };
                    })
                    .ToArray();

            var document =
                new
                {
                    Schema =
                        "pipeline-v2-target-coverage-v1",
                    SourceFile =
                        Path.GetFileName(
                            sourcePath),
                    DetectedTextTargets =
                        targets.Length,
                    CommittedTargets =
                        targets.Count(x =>
                            x.Status ==
                            "translated_committed"),
                    PreservedTargets =
                        targets.Count(x =>
                            x.Status.StartsWith(
                                "preserved_",
                                StringComparison.Ordinal)),
                    UnresolvedTargets =
                        targets.Count(x =>
                            x.Unresolved),
                    CoverageComplete =
                        targets.All(x =>
                            !x.Unresolved),
                    Targets =
                        targets
                };

            File.WriteAllText(
                path,
                JsonSerializer.Serialize(
                    document,
                    new JsonSerializerOptions
                    {
                        WriteIndented =
                            true
                    }));
        }
        catch
        {
            // Coverage diagnostics must never invalidate a finished page.
        }
    }

    static double TargetAffinityScore(
        VisionTranslation region,
        V2TextTarget target)
    {
        if (region.Source.RegionTextRegion is { } linked &&
            string.Equals(
                linked.RegionId,
                target.TextRegionId,
                StringComparison.Ordinal))
        {
            return 1000;
        }

        if (!string.IsNullOrWhiteSpace(
                region.Source.RegionId) &&
            !string.IsNullOrWhiteSpace(
                target.BubbleRegionId) &&
            string.Equals(
                region.Source.RegionId,
                target.BubbleRegionId,
                StringComparison.Ordinal))
        {
            return 600;
        }

        var source =
            new Rect2d(
                region.Source.X,
                region.Source.Y,
                Math.Max(
                    1.0,
                    region.Source.W),
                Math.Max(
                    1.0,
                    region.Source.H));

        double overlap =
            IntersectionArea(
                source,
                new Rect2d(
                    target.TextBounds.X,
                    target.TextBounds.Y,
                    target.TextBounds.Width,
                    target.TextBounds.Height));

        if (overlap <= 0)
            return 0;

        double sourceArea =
            Math.Max(
                1.0,
                source.Width *
                source.Height);

        double targetArea =
            Math.Max(
                1.0,
                target.TextBounds.Width *
                (double)target.TextBounds.Height);

        double containment =
            overlap /
            Math.Min(
                sourceArea,
                targetArea);

        return containment >= 0.18
            ? 100 + containment * 100
            : 0;
    }

    static bool IsLikelyAuditOcrNoise(
        VisionTranslation region)
    {
        string source =
            region.Source.Text.Trim();

        int meaningful =
            source.Count(
                char.IsLetterOrDigit);

        if (meaningful == 0)
            return true;

        if (meaningful > 1)
            return false;

        return region.Source.Lines.Count <= 2;
    }

    static Dictionary<int, BalloonLayout> AnalyzeContainers(
        string sourcePath,
        IReadOnlyList<VisionTranslation> regions,
        IProgress<string>? progress,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidOperationException("원본 이미지를 열 수 없습니다.");

        var result = new Dictionary<int, BalloonLayout>(regions.Count);

        foreach (var region in regions)
        {
            token.ThrowIfCancellationRequested();

            var layout =
                region.Source.RegionContainer is { } regionContainer
                    ? BalloonMaskService.AnalyzeConfirmedRegion(
                        source,
                        region.Source,
                        region.Type,
                        regionContainer)
                    : BalloonMaskService.Analyze(
                        source,
                        region.Source,
                        region.Type);

            result[region.Id] = layout;
            progress?.Report(layout.Diagnostic(region.Id));
        }

        return result;
    }

    static List<RenderUnitPlan> BuildRenderPlans(
        IReadOnlyList<VisionTranslation> regions,
        IReadOnlyDictionary<int, BalloonLayout> layouts,
        out List<VisionTranslation> containerSuppressed,
        IProgress<string>? progress,
        bool requireLegacyErase = true)
    {
        containerSuppressed = [];

        // 더 완전하고 신뢰할 수 있는 후보가 동일 물리 컨테이너의 소유권을 먼저 얻는다.
        var ordered = regions
            .OrderByDescending(x => ContainerKeepScore(
                x,
                layouts.TryGetValue(x.Id, out var l) ? l : null))
            .ThenBy(x => x.Id)
            .ToList();

        var selected = new List<ContainerSelection>();
        int nextContainer = 1;

        foreach (var region in ordered)
        {
            if (!layouts.TryGetValue(region.Id, out var layout))
                continue;

            // Legacy rendering still suppresses duplicate physical containers
            // up front.  Pipeline V2 must not: 0033_2 showed that a weak OCR
            // candidate can be selected first, fail semantic validation, and
            // permanently suppress a later exact/secondary-backed candidate
            // (page032 AM I RIGHT?).  V2 evaluates every candidate first and
            // arbitrates only among candidates that are actually renderable.
            if (requireLegacyErase &&
                layout.Detected)
            {
                var same =
                    selected.FirstOrDefault(x =>
                        x.Layout.Detected &&
                        IsSamePhysicalContainer(
                            x.Layout,
                            layout));

                if (same is not null)
                {
                    containerSuppressed.Add(
                        region);

                    continue;
                }
            }

            selected.Add(new ContainerSelection(
                region,
                layout,
                $"C{nextContainer++:000}"));
        }

        selected = selected
            .OrderBy(x => x.Region.Id)
            .ToList();

        var linePool = new List<CanonicalLine>();
        var lineOwners = new Dictionary<string, string>();
        int nextLine = 1;
        int nextUnit = 1;

        double pageFontReference = ComputePageFontReference(
            selected.Select(x => x.Region).ToList());

        var plans = new List<RenderUnitPlan>(selected.Count);

        foreach (var item in selected)
        {
            string unitId = $"U{nextUnit++:000}";
            var plannedLines = new List<PlannedOcrLine>();

            foreach (var line in item.Region.Source.Lines)
            {
                var canonical = linePool.FirstOrDefault(x =>
                    IsSameCanonicalLine(x.Line, line));

                if (canonical is null)
                {
                    canonical = new CanonicalLine(
                        $"L{nextLine++:0000}",
                        line);
                    linePool.Add(canonical);
                }

                plannedLines.Add(new PlannedOcrLine(
                    canonical.LineId,
                    line));
            }

            var conflicts =
                requireLegacyErase
                    ? plannedLines
                        .Where(x =>
                            lineOwners.ContainsKey(
                                x.LineId))
                        .Select(x =>
                            x.LineId)
                        .Distinct(
                            StringComparer.Ordinal)
                        .ToList()
                    : [];

            // Canonical-line ownership is a legacy erase safety rule.  V2
            // erase ownership is the immutable RT-DETR TextRegion, so two OCR
            // candidates for the same detector target must both survive long
            // enough for evidence arbitration.
            var ownedLines =
                requireLegacyErase
                    ? plannedLines
                        .Where(x =>
                            !lineOwners.ContainsKey(
                                x.LineId))
                        .ToList()
                    : plannedLines.ToList();

            if (conflicts.Count > 0 &&
                ownedLines.Count > 0)
            {
                progress?.Report(
                    $"[line-trim] {unitId}/{item.ContainerId} id={item.Region.Id} " +
                    $"duplicate={string.Join("+", conflicts)}");
            }

            var erase = CreateErasePlan(
                unitId,
                item.ContainerId,
                item.Region,
                item.Layout,
                ownedLines);

            var textLayout = CreateTextLayoutPlan(
                item.Region,
                item.Layout,
                pageFontReference);

            bool semanticSafe =
                !RenderSafetyPolicy.IsSuspiciousVisionExpansion(
                    item.Region);

            bool allLinesConflict =
                plannedLines.Count > 0 &&
                ownedLines.Count == 0 &&
                conflicts.Count > 0;

            bool approved = item.Layout.ShouldRender &&
                            semanticSafe &&
                            textLayout.Fits &&
                            !allLinesConflict &&
                            (!requireLegacyErase ||
                             erase.Approved);

            string reason;
            if (!item.Layout.ShouldRender)
                reason = $"container_rejected:{item.Layout.Reason}";
            else if (!semanticSafe)
                reason = "vision_expansion_untrusted";
            else if (allLinesConflict)
                reason = $"line_ownership_conflict_all:{string.Join("+", conflicts)}";
            else if (!textLayout.Fits)
                reason = $"layout_rejected:{textLayout.Reason}";
            else if (requireLegacyErase &&
                     !erase.Approved)
                reason = $"erase_rejected:{erase.Reason}";
            else
                reason = requireLegacyErase
                    ? "ok"
                    : "ok_v2_erase";

            if (approved &&
                requireLegacyErase)
            {
                // Only legacy rendering claims canonical OCR lines here.
                // V2 arbitrates candidates by detector-owned physical target
                // after semantic/layout validation.
                foreach (var line in ownedLines)
                    lineOwners[line.LineId] = unitId;
            }
            else
            {
                progress?.Report(
                    $"[plan-keep] {unitId}/{item.ContainerId} id={item.Region.Id} reason={reason}");
            }

            plans.Add(new RenderUnitPlan(
                unitId,
                item.ContainerId,
                item.Region,
                item.Layout,
                plannedLines,
                erase,
                textLayout,
                approved,
                reason));
        }

        if (!requireLegacyErase)
        {
            var approved =
                plans
                    .Where(x =>
                        x.Approved)
                    .OrderByDescending(x =>
                        V2CandidateKeepScore(
                            x.Region))
                    .ThenBy(x =>
                        x.Region.Id)
                    .ToList();

            var kept =
                new List<RenderUnitPlan>();

            var suppressedRegionIds =
                new HashSet<int>();

            foreach (var candidate in approved)
            {
                bool duplicate =
                    kept.Any(existing =>
                        existing.Container.Detected &&
                        candidate.Container.Detected &&
                        IsSamePhysicalContainer(
                            existing.Container,
                            candidate.Container));

                if (!duplicate)
                {
                    kept.Add(
                        candidate);

                    continue;
                }

                suppressedRegionIds.Add(
                    candidate.Region.Id);

                containerSuppressed.Add(
                    candidate.Region);
            }

            if (suppressedRegionIds.Count > 0)
            {
                plans =
                    plans
                        .Select(plan =>
                            suppressedRegionIds.Contains(
                                plan.Region.Id)
                                ? plan with
                                {
                                    Approved = false,
                                    Reason =
                                        "container_duplicate_after_v2_candidate_arbitration"
                                }
                                : plan)
                        .ToList();
            }
        }

        return plans;
    }

    static double V2CandidateKeepScore(
        VisionTranslation region)
    {
        double score =
            DuplicateKeepScore(
                region);

        if (region.Source.RegionTextRegion is not null)
            score += 40000;

        if (!string.IsNullOrWhiteSpace(
                region.Source.RegionId))
        {
            score += 20000;
        }

        string agreement =
            region.Source.SecondaryOcrAgreement ??
            "";

        if (string.Equals(
                agreement,
                "agree",
                StringComparison.OrdinalIgnoreCase))
        {
            score += 50000;
        }
        else if (string.Equals(
                     agreement,
                     "partial",
                     StringComparison.OrdinalIgnoreCase))
        {
            score += 20000;
        }
        else if (string.Equals(
                     agreement,
                     "secondary_only",
                     StringComparison.OrdinalIgnoreCase))
        {
            score += 15000;
        }

        if (region.Source.Lines.Count > 0)
        {
            double confidence =
                region.Source.Lines.Average(x =>
                    Math.Clamp(
                        x.Confidence,
                        0,
                        1));

            score +=
                confidence *
                25000;
        }

        if (RenderSafetyPolicy.IsSuspiciousVisionExpansion(
                region))
        {
            score -= 100000;
        }

        return score;
    }

    static ErasePlan CreateErasePlan(
        string unitId,
        string containerId,
        VisionTranslation region,
        BalloonLayout layout,
        IReadOnlyList<PlannedOcrLine> lines)
    {
        int width = Math.Max(1, layout.Bounds.Width);
        int height = Math.Max(1, layout.Bounds.Height);
        var empty = new byte[width * height];

        // A fallback rectangle is useful for layout, but it is not proof that
        // the pixels inside it are source text. 0009 therefore refuses erase
        // permission unless a real container and its SafeMask both exist.
        if (!layout.Detected)
        {
            return new ErasePlan(
                unitId,
                containerId,
                false,
                "container_not_detected",
                layout.Bounds,
                empty,
                width,
                height,
                []);
        }

        if (layout.SafeMask is not { Length: > 0 } ||
            layout.MaskWidth != width ||
            layout.MaskHeight != height)
        {
            return new ErasePlan(
                unitId,
                containerId,
                false,
                "safe_mask_missing",
                layout.Bounds,
                empty,
                width,
                height,
                []);
        }

        var approvedLineIds = lines
            .Where(x => IsEraseEligibleLine(
                x.Source,
                layout))
            .Select(x => x.LineId)
            .Distinct(StringComparer.Ordinal)
            .ToList();

        bool relaxed = false;

        // 0010 showed that RT-DETR can correctly confirm a speech/caption
        // container while the primary OCR line narrowly misses one of the
        // conservative erase gates. Only relax when the same region also has
        // independent secondary OCR evidence. This keeps artwork-only false
        // positives on the strict path.
        if (approvedLineIds.Count == 0 &&
            HasTrustedContainerEvidence(region, layout))
        {
            approvedLineIds = lines
                .Where(x => IsRelaxedEraseEligibleLine(
                    x.Source,
                    layout))
                .Select(x => x.LineId)
                .Distinct(StringComparer.Ordinal)
                .ToList();

            relaxed = approvedLineIds.Count > 0;
        }

        byte[]? textRegionMask =
            null;

        bool learnedTextRegion =
            false;

        if (approvedLineIds.Count == 0 &&
            HasTrustedTextRegionEvidence(
                region,
                layout))
        {
            approvedLineIds =
                lines
                    .Where(x =>
                        IsTextRegionEraseEligibleLine(
                            x.Source,
                            region.Source.RegionTextRegion!))
                    .Select(x =>
                        x.LineId)
                    .Distinct(
                        StringComparer.Ordinal)
                    .ToList();

            if (approvedLineIds.Count > 0)
            {
                textRegionMask =
                    BuildTextRegionAllowedMask(
                        layout,
                        region.Source.RegionTextRegion!);

                learnedTextRegion =
                    textRegionMask.Any(x =>
                        x != 0);

                if (!learnedTextRegion)
                    approvedLineIds = [];
            }
        }

        if (approvedLineIds.Count == 0)
        {
            return new ErasePlan(
                unitId,
                containerId,
                false,
                "no_safe_text_line",
                layout.Bounds,
                empty,
                width,
                height,
                []);
        }

        return new ErasePlan(
            unitId,
            containerId,
            true,
            learnedTextRegion
                ? "safe_rtdetr_text_region"
                : relaxed
                    ? "safe_container_text_relaxed"
                    : "safe_container_text",
            layout.Bounds,
            learnedTextRegion
                ? textRegionMask!
                : layout.SafeMask.ToArray(),
            width,
            height,
            approvedLineIds);
    }

    static bool IsEraseEligibleLine(
        OcrLine line,
        BalloonLayout layout)
    {
        if (!layout.Detected)
            return false;

        double coverage = LineMaskCoverage(
            line,
            layout);

        if (coverage < 0.45)
            return false;

        string text = line.Text.Trim();
        int meaningful = text.Count(char.IsLetterOrDigit);
        if (meaningful < 2)
            return false;

        if (line.Confidence < 0.55f)
            return false;

        double containerArea = Math.Max(
            1.0,
            layout.Bounds.Width * (double)layout.Bounds.Height);

        double lineArea = Math.Max(
            1.0,
            line.W * line.H);

        // Very large OCR boxes over faces, fists or textured artwork are a
        // recurring false-positive pattern. Real dialogue lines normally occupy
        // only a narrow strip of a speech/caption container.
        if (lineArea / containerArea > 0.28)
            return false;

        if (line.H > layout.Bounds.Height * 0.48)
            return false;

        return true;
    }

    static bool HasTrustedContainerEvidence(
        VisionTranslation region,
        BalloonLayout layout)
    {
        if (!layout.Detected ||
            region.Source.RegionContainer is not { } candidate ||
            string.IsNullOrWhiteSpace(candidate.RegionId) ||
            string.IsNullOrWhiteSpace(region.Source.SecondaryOcrText))
        {
            return false;
        }

        int secondaryMeaningful =
            region.Source.SecondaryOcrText.Count(
                char.IsLetterOrDigit);

        return secondaryMeaningful >= 2 &&
               candidate.Score >= 3.0;
    }

    static bool IsRelaxedEraseEligibleLine(
        OcrLine line,
        BalloonLayout layout)
    {
        if (!layout.Detected)
            return false;

        double coverage = LineMaskCoverage(
            line,
            layout);

        if (coverage < 0.30)
            return false;

        int meaningful =
            line.Text.Trim().Count(
                char.IsLetterOrDigit);

        if (meaningful < 1 ||
            line.Confidence < 0.35f)
        {
            return false;
        }

        double containerArea = Math.Max(
            1.0,
            layout.Bounds.Width * (double)layout.Bounds.Height);

        double lineArea = Math.Max(
            1.0,
            line.W * line.H);

        if (lineArea / containerArea > 0.42)
            return false;

        if (line.H > layout.Bounds.Height * 0.65)
            return false;

        return true;
    }

    static bool HasTrustedTextRegionEvidence(
        VisionTranslation region,
        BalloonLayout layout)
    {
        if (!layout.Detected ||
            region.Source.RegionContainer is not { } candidate ||
            region.Source.RegionTextRegion is not { } textRegion ||
            textRegion.Kind != PageRegionKind.TextBubble ||
            textRegion.Score < 0.72f)
        {
            return false;
        }

        var bubbleBounds =
            candidate.LearnedBounds ??
            candidate.Bounds;

        double textInsideBubble =
            IntersectionArea(
                textRegion.Bounds,
                bubbleBounds) /
            Math.Max(
                1.0,
                textRegion.Bounds.Width *
                (double)textRegion.Bounds.Height);

        return textInsideBubble >= 0.60;
    }

    static bool IsTextRegionEraseEligibleLine(
        OcrLine line,
        PageRegion textRegion)
    {
        int meaningful =
            line.Text.Trim().Count(
                char.IsLetterOrDigit);

        if (meaningful < 1 ||
            line.Confidence < 0.30f)
        {
            return false;
        }

        var lineRect =
            new CvRect(
                (int)Math.Floor(
                    line.X),
                (int)Math.Floor(
                    line.Y),
                Math.Max(
                    1,
                    (int)Math.Ceiling(
                        line.W)),
                Math.Max(
                    1,
                    (int)Math.Ceiling(
                        line.H)));

        double coverage =
            IntersectionArea(
                lineRect,
                textRegion.Bounds) /
            Math.Max(
                1.0,
                lineRect.Width *
                (double)lineRect.Height);

        return coverage >= 0.40;
    }

    static byte[] BuildTextRegionAllowedMask(
        BalloonLayout layout,
        PageRegion textRegion)
    {
        int width =
            Math.Max(
                1,
                layout.Bounds.Width);

        int height =
            Math.Max(
                1,
                layout.Bounds.Height);

        var mask =
            new byte[
                width *
                height];

        int left =
            Math.Max(
                layout.Bounds.Left,
                textRegion.Bounds.Left);

        int top =
            Math.Max(
                layout.Bounds.Top,
                textRegion.Bounds.Top);

        int right =
            Math.Min(
                layout.Bounds.Right,
                textRegion.Bounds.Right);

        int bottom =
            Math.Min(
                layout.Bounds.Bottom,
                textRegion.Bounds.Bottom);

        if (right <= left ||
            bottom <= top)
        {
            return mask;
        }

        // Keep a tiny detector-edge guard while still allowing the OCR glyph
        // dilation to cover anti-aliased outlines. The final inpaint mask is
        // clipped to this RT-DETR TextBubble rectangle.
        int guard =
            Math.Clamp(
                Math.Min(
                    right - left,
                    bottom - top) / 80,
                0,
                2);

        left += guard;
        top += guard;
        right -= guard;
        bottom -= guard;

        if (right <= left ||
            bottom <= top)
        {
            return mask;
        }

        for (int y = top;
             y < bottom;
             y++)
        {
            int localY =
                y -
                layout.Bounds.Y;

            if (localY < 0 ||
                localY >= height)
            {
                continue;
            }

            for (int x = left;
                 x < right;
                 x++)
            {
                int localX =
                    x -
                    layout.Bounds.X;

                if (localX < 0 ||
                    localX >= width)
                {
                    continue;
                }

                mask[
                    localY * width +
                    localX] =
                    255;
            }
        }

        return mask;
    }

    static TextLayoutPlan CreateTextLayoutPlan(
        VisionTranslation region,
        BalloonLayout layout,
        double pageFontReference)
    {
        string text = NormalizeForAutoLayout(region.Translation);
        if (string.IsNullOrWhiteSpace(text))
            return FailedLayout("empty_translation", text, layout.Inner);

        if (!layout.ShouldRender)
            return FailedLayout("container_rejected", text, layout.Inner);

        var box = new WpfRect(
            layout.Inner.X,
            layout.Inner.Y,
            Math.Max(12, layout.Inner.Width),
            Math.Max(12, layout.Inner.Height));

        var culture = CultureInfo.GetCultureInfo("ko-KR");
        var typeface = CreateTypeface();

        bool caption = string.Equals(
            region.Type,
            "caption",
            StringComparison.OrdinalIgnoreCase);

        TextAlignment alignment = caption
            ? TextAlignment.Left
            : TextAlignment.Center;

        double minFont = Math.Clamp(
            Math.Min(box.Width, box.Height) * 0.055,
            8,
            13);

        int meaningfulTranslation =
            text.Count(char.IsLetterOrDigit);

        bool shortWideCaption =
            caption &&
            meaningfulTranslation > 0 &&
            meaningfulTranslation <= 24 &&
            box.Width >= box.Height * 3.0;

        double containerMaxFont = Math.Clamp(
            caption
                ? Math.Min(
                    box.Height *
                    (shortWideCaption
                        ? 0.46
                        : 0.34),
                    shortWideCaption
                        ? 52
                        : 48)
                : Math.Min(box.Height * 0.46, 56),
            Math.Max(minFont, 12),
            caption
                ? shortWideCaption
                    ? 52
                    : 48
                : 56);

        double sourceGlyphHint = ComputeSourceGlyphHint(
            region.Source,
            pageFontReference);

        double normalizedGlyph = Math.Clamp(
            sourceGlyphHint,
            pageFontReference * 0.78,
            pageFontReference * 1.22);

        // Short single-line sign/caption text such as
        // "ALICE - FOUR DAYS UNTIL ENTRY" already owns a wide parent
        // rectangle. Keeping it at 0.92x the OCR glyph hint leaves the Korean
        // translation looking like a tiny label inside that box. Let those
        // captions use more of the confirmed parent geometry, while dense
        // narration keeps the conservative scale.
        double sourceDrivenMax =
            normalizedGlyph *
            (caption
                ? shortWideCaption
                    ? 1.18
                    : 0.92
                : 0.98);

        double maxFont = Math.Min(
            containerMaxFont,
            Math.Clamp(
                sourceDrivenMax,
                Math.Max(minFont, 12),
                caption ? 46 : 52));

        double[] lineHeightFactors = [1.12, 1.06, 1.00];

        TextLayoutPlan? best = null;

        foreach (double factor in lineHeightFactors)
        {
            double low = Math.Min(minFont, maxFont);
            double high = Math.Max(minFont, maxFont);
            TextLayoutPlan? localBest = null;

            for (int i = 0; i < 13; i++)
            {
                double size = (low + high) / 2.0;
                var candidate = EvaluateLayout(
                    text,
                    culture,
                    typeface,
                    size,
                    factor,
                    alignment,
                    box,
                    caption,
                    shortWideCaption);

                if (candidate.Fits)
                {
                    localBest = candidate;
                    low = size;
                }
                else
                {
                    high = size;
                }
            }

            if (localBest is null)
                continue;

            if (best is null ||
                LayoutReadabilityScore(localBest) > LayoutReadabilityScore(best))
            {
                best = localBest;
            }
        }

        if (best is not null)
            return best;

        // 박스를 키우는 대신 가독성 하한까지만 폰트를 줄인다.
        for (double size = Math.Min(minFont, maxFont); size >= 6.0; size -= 0.5)
        {
            var candidate = EvaluateLayout(
                text,
                culture,
                typeface,
                size,
                1.0,
                alignment,
                box,
                caption,
                shortWideCaption);

            if (candidate.Fits)
                return candidate;
        }

        return FailedLayout(
            "overflow_at_min_font",
            text,
            layout.Inner);
    }

    static double LayoutReadabilityScore(TextLayoutPlan plan)
        => plan.FontSize -
           (1.12 - plan.LineHeightFactor) * 12.0;

    static TextLayoutPlan EvaluateLayout(
        string text,
        CultureInfo culture,
        Typeface typeface,
        double fontSize,
        double lineHeightFactor,
        TextAlignment alignment,
        WpfRect box,
        bool caption,
        bool centerShortWideCaption)
    {
        var formatted = MakeFormattedText(
            text,
            culture,
            typeface,
            fontSize,
            box.Width,
            lineHeightFactor,
            alignment);

        double originY =
            caption &&
            !centerShortWideCaption
                ? box.Y
                : box.Y +
                  Math.Max(
                      0,
                      (box.Height -
                       formatted.Height) /
                      2.0);

        double originX = box.X;
        var origin = new System.Windows.Point(originX, originY);

        double outline = Math.Clamp(
            fontSize * 0.035,
            0.45,
            1.25);

        var geometry = formatted.BuildGeometry(origin);
        var glyph = geometry.Bounds;
        glyph.Inflate(outline * 0.6 + 0.5, outline * 0.6 + 0.5);

        bool fits =
            formatted.Height <= box.Height * 0.985 &&
            glyph.Left >= box.Left - 0.5 &&
            glyph.Top >= box.Top - 0.5 &&
            glyph.Right <= box.Right + 0.5 &&
            glyph.Bottom <= box.Bottom + 0.5;

        return new TextLayoutPlan(
            fits,
            fits ? "ok" : "glyph_outside_safe_area",
            text,
            box.X,
            box.Y,
            box.Width,
            box.Height,
            fontSize,
            lineHeightFactor,
            alignment == TextAlignment.Left ? "left" : "center",
            originX,
            originY,
            glyph.X,
            glyph.Y,
            glyph.Width,
            glyph.Height,
            outline);
    }

    static TextLayoutPlan FailedLayout(
        string reason,
        string text,
        CvRect inner)
        => new(
            false,
            reason,
            text,
            inner.X,
            inner.Y,
            Math.Max(1, inner.Width),
            Math.Max(1, inner.Height),
            0,
            1.0,
            "center",
            inner.X,
            inner.Y,
            0,
            0,
            0,
            0,
            0);

    static HashSet<string> InpaintApprovedUnits(
        string sourcePath,
        string cleanedPath,
        string eraseDebugPath,
        IReadOnlyList<RenderUnitPlan> plans,
        IProgress<string>? progress,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
        if (source.Empty())
            throw new InvalidOperationException("원본 이미지를 열 수 없습니다.");

        using var globalMask = Mat.Zeros(
            source.Rows,
            source.Cols,
            MatType.CV_8UC1).ToMat();

        var erasedUnits = new HashSet<string>(
            StringComparer.Ordinal);

        foreach (var plan in plans.Where(x =>
                     x.Approved &&
                     x.Erase.Approved))
        {
            token.ThrowIfCancellationRequested();

            using var allowed = Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1).ToMat();

            MaterializeAllowedMask(
                allowed,
                plan.Erase);

            using var textMask = Mat.Zeros(
                source.Rows,
                source.Cols,
                MatType.CV_8UC1).ToMat();

            foreach (var line in plan.Lines.Where(x =>
                         plan.Erase.LineIds.Contains(
                             x.LineId,
                             StringComparer.Ordinal)))
            {
                token.ThrowIfCancellationRequested();

                AddTextCandidateMask(
                    source,
                    textMask,
                    allowed,
                    line.Source);
            }

            // 고정 3x3 팽창은 굵은 만화 폰트의 외곽선/안티앨리어싱을
            // 충분히 덮지 못했다. OCR 글자 크기에 비례해 삭제 마스크를 확장하되,
            // 아래에서 반드시 SafeMask와 다시 교차해 말풍선 밖으로는 나가지 않는다.
            int dilationRadius = ComputeEraseDilationRadius(plan);

            using var expanded = new Mat();
            using (var kernel = Cv2.GetStructuringElement(
                MorphShapes.Ellipse,
                new OpenCvSharp.Size(
                    dilationRadius * 2 + 1,
                    dilationRadius * 2 + 1)))
            {
                Cv2.Dilate(
                    textMask,
                    expanded,
                    kernel,
                    iterations: 1);
            }

            // 핵심 불변조건:
            // 팽창 후에 반드시 승인된 삭제 허용영역과 다시 교차시킨다.
            using var finalUnitMask = new Mat();
            Cv2.BitwiseAnd(
                expanded,
                allowed,
                finalUnitMask);

            int pixels = Cv2.CountNonZero(finalUnitMask);
            if (pixels < 4)
            {
                progress?.Report(
                    $"[erase-keep] {plan.UnitId}/{plan.ContainerId} id={plan.Region.Id} mask_pixels={pixels}");
                continue;
            }

            Cv2.BitwiseOr(
                globalMask,
                finalUnitMask,
                globalMask);

            erasedUnits.Add(plan.UnitId);
        }

        SaveEraseDebug(
            source,
            globalMask,
            eraseDebugPath);

        token.ThrowIfCancellationRequested();

        if (Cv2.CountNonZero(globalMask) == 0)
        {
            if (!Cv2.ImWrite(cleanedPath, source))
                throw new InvalidOperationException("원문 보존 이미지를 저장하지 못했습니다.");

            return erasedUnits;
        }

        using var inpainted = new Mat();
        Cv2.Inpaint(
            source,
            globalMask,
            inpainted,
            InpaintRadius,
            InpaintTypes.Telea);

        // 인페인터가 반환한 전체 이미지를 채택하지 않는다.
        // 승인된 최종 삭제 마스크 픽셀만 원본에 다시 합성한다.
        using var cleaned = source.Clone();
        inpainted.CopyTo(
            cleaned,
            globalMask);

        if (!Cv2.ImWrite(cleanedPath, cleaned))
            throw new InvalidOperationException("인페인트 결과 이미지를 저장하지 못했습니다.");

        return erasedUnits;
    }

    static void MaterializeAllowedMask(
        Mat destination,
        ErasePlan plan)
    {
        var bounds = ClampRect(
            plan.AllowedBounds.X,
            plan.AllowedBounds.Y,
            plan.AllowedBounds.Width,
            plan.AllowedBounds.Height,
            destination.Cols,
            destination.Rows);

        if (bounds.Width <= 0 || bounds.Height <= 0)
            return;

        int srcWidth = Math.Max(1, plan.MaskWidth);
        int srcHeight = Math.Max(1, plan.MaskHeight);

        for (int y = 0; y < bounds.Height; y++)
        {
            int srcY = y + (bounds.Y - plan.AllowedBounds.Y);
            if (srcY < 0 || srcY >= srcHeight)
                continue;

            for (int x = 0; x < bounds.Width; x++)
            {
                int srcX = x + (bounds.X - plan.AllowedBounds.X);
                if (srcX < 0 || srcX >= srcWidth)
                    continue;

                int index = srcY * srcWidth + srcX;
                if (index >= 0 &&
                    index < plan.AllowedMask.Length &&
                    plan.AllowedMask[index] != 0)
                {
                    destination.Set(
                        bounds.Y + y,
                        bounds.X + x,
                        (byte)255);
                }
            }
        }
    }

    static int ComputeEraseDilationRadius(
        RenderUnitPlan plan)
    {
        var sizes = plan.Lines
            .Select(x => Math.Max(
                1.0,
                Math.Min(
                    x.Source.W,
                    x.Source.H)))
            .OrderBy(x => x)
            .ToArray();

        double glyphSize;

        if (sizes.Length == 0)
        {
            glyphSize = 24;
        }
        else
        {
            int mid = sizes.Length / 2;
            glyphSize = sizes.Length % 2 == 1
                ? sizes[mid]
                : (sizes[mid - 1] + sizes[mid]) / 2.0;
        }

        bool caption = string.Equals(
            plan.Region.Type,
            "caption",
            StringComparison.OrdinalIgnoreCase);

        int maxRadius = caption
            ? 5
            : 7;

        return (int)Math.Clamp(
            Math.Round(glyphSize * 0.10),
            2,
            maxRadius);
    }

    static void AddTextCandidateMask(
        Mat source,
        Mat textMask,
        Mat allowed,
        OcrLine line)
    {
        double glyphScale = Math.Max(
            1.0,
            Math.Min(line.W, line.H));

        // OCR box 가장자리 밖으로 삐져나온 serif/outline까지 후보 검사에 포함한다.
        // 실제 삭제는 allowed(SafeMask)로 다시 제한되므로 컨테이너 밖 픽셀은 지워지지 않는다.
        int expand = (int)Math.Clamp(
            Math.Ceiling(glyphScale * 0.10),
            2,
            10);

        var textRect = ClampRect(
            (int)Math.Floor(line.X) - expand,
            (int)Math.Floor(line.Y) - expand,
            (int)Math.Ceiling(line.W) + expand * 2,
            (int)Math.Ceiling(line.H) + expand * 2,
            source.Cols,
            source.Rows);

        int ringPad = (int)Math.Clamp(
            glyphScale * 0.45,
            5,
            20);

        var sampleRect = ClampRect(
            textRect.X - ringPad,
            textRect.Y - ringPad,
            textRect.Width + ringPad * 2,
            textRect.Height + ringPad * 2,
            source.Cols,
            source.Rows);

        var samples = CollectAllowedRingSamples(
            source,
            allowed,
            sampleRect,
            textRect);

        if (samples.Count < 12)
        {
            for (int y = sampleRect.Top; y < sampleRect.Bottom; y += 2)
            {
                for (int x = sampleRect.Left; x < sampleRect.Right; x += 2)
                {
                    if (allowed.At<byte>(y, x) == 0)
                        continue;

                    samples.Add(source.At<Vec3b>(y, x));
                }
            }
        }

        if (samples.Count == 0)
            return;

        var background = MedianColor(samples);

        var distances = samples
            .Select(p => ColorDistance(p, background))
            .OrderBy(x => x)
            .ToList();

        double naturalVariation = Percentile(
            distances,
            0.82);

        // 0003에서 옅은 외곽선이 남는 사례가 있어 seed 검출을 약간 넓힌다.
        // 넓은 영역을 지우는 방식이 아니라 OCR 주변의 실제 색/명도 차이 픽셀만 추가한다.
        double threshold = Math.Clamp(
            naturalVariation + 12,
            22,
            66);

        int marked = 0;
        int allowedPixels = 0;
        double bgLuma = Luminance(background);

        for (int y = textRect.Top; y < textRect.Bottom; y++)
        {
            for (int x = textRect.Left; x < textRect.Right; x++)
            {
                if (allowed.At<byte>(y, x) == 0)
                    continue;

                allowedPixels++;

                var pixel = source.At<Vec3b>(y, x);
                double distance = ColorDistance(
                    pixel,
                    background);

                double lumaDelta = Math.Abs(
                    Luminance(pixel) - bgLuma);

                if (distance >= threshold ||
                    lumaDelta >= Math.Max(22, threshold * 0.70))
                {
                    textMask.Set(y, x, (byte)255);
                    marked++;
                }
            }
        }

        // 일부 획만 잡힌 경우에도 옅은 안티앨리어싱이 남지 않도록 보강한다.
        if (allowedPixels > 0 &&
            marked / (double)allowedPixels < 0.045)
        {
            for (int y = textRect.Top; y < textRect.Bottom; y++)
            {
                for (int x = textRect.Left; x < textRect.Right; x++)
                {
                    if (allowed.At<byte>(y, x) == 0)
                        continue;

                    var pixel = source.At<Vec3b>(y, x);
                    double delta =
                        Luminance(pixel) - bgLuma;

                    bool likelyText = bgLuma >= 145
                        ? delta <= -28
                        : bgLuma <= 110
                            ? delta >= 28
                            : Math.Abs(delta) >= 34;

                    if (likelyText)
                        textMask.Set(y, x, (byte)255);
                }
            }
        }
    }

    static List<Vec3b> CollectAllowedRingSamples(
        Mat source,
        Mat allowed,
        CvRect outer,
        CvRect inner)
    {
        var samples = new List<Vec3b>();

        for (int y = outer.Top; y < outer.Bottom; y += 2)
        {
            for (int x = outer.Left; x < outer.Right; x += 2)
            {
                if (allowed.At<byte>(y, x) == 0)
                    continue;

                if (x >= inner.Left &&
                    x < inner.Right &&
                    y >= inner.Top &&
                    y < inner.Bottom)
                    continue;

                samples.Add(
                    source.At<Vec3b>(y, x));
            }
        }

        return samples;
    }

    static void TypesetAndSave(
        string cleanedPath,
        IReadOnlyList<RenderUnitPlan> plans,
        string outputPath,
        CancellationToken token)
    {
        var bitmap = LoadBitmap(cleanedPath);
        int width = bitmap.PixelWidth;
        int height = bitmap.PixelHeight;

        var visual = new DrawingVisual();

        using (var dc = visual.RenderOpen())
        {
            dc.DrawImage(
                bitmap,
                new WpfRect(0, 0, width, height));

            foreach (var plan in plans
                         .OrderBy(x => x.Region.Id))
            {
                token.ThrowIfCancellationRequested();

                var lp = plan.Layout;
                if (!lp.Fits ||
                    string.IsNullOrWhiteSpace(lp.Text))
                    continue;

                var culture = CultureInfo.GetCultureInfo("ko-KR");
                var typeface = CreateTypeface();
                TextAlignment alignment =
                    string.Equals(lp.Alignment, "left", StringComparison.OrdinalIgnoreCase)
                        ? TextAlignment.Left
                        : TextAlignment.Center;

                var formatted = MakeFormattedText(
                    lp.Text,
                    culture,
                    typeface,
                    lp.FontSize,
                    lp.BoxWidth,
                    lp.LineHeightFactor,
                    alignment);

                var origin = new System.Windows.Point(
                    lp.OriginX,
                    lp.OriginY);

                var geometry = formatted.BuildGeometry(
                    origin);

                var box = new WpfRect(
                    lp.BoxX,
                    lp.BoxY,
                    lp.BoxWidth,
                    lp.BoxHeight);

                // 검증을 통과했더라도 최종 렌더러는 안전영역 clip을 마지막 방어선으로 둔다.
                dc.PushClip(
                    new RectangleGeometry(box));

                var pen = new Pen(
                    Brushes.White,
                    lp.OutlineWidth);

                dc.DrawGeometry(
                    Brushes.Black,
                    pen,
                    geometry);

                dc.Pop();
            }
        }

        var rendered = new RenderTargetBitmap(
            width,
            height,
            96,
            96,
            PixelFormats.Pbgra32);

        rendered.Render(visual);
        rendered.Freeze();

        Directory.CreateDirectory(
            Path.GetDirectoryName(outputPath)!);

        string tempPng = Path.Combine(
            Path.GetTempPath(),
            $"lmt_typeset_{Guid.NewGuid():N}.png");

        try
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(
                BitmapFrame.Create(rendered));

            using (var stream = File.Create(tempPng))
                encoder.Save(stream);

            using var final = Cv2.ImRead(
                tempPng,
                ImreadModes.Color);

            if (final.Empty() ||
                !SaveLosslessWebp(
                    outputPath,
                    final))
            {
                throw new InvalidOperationException(
                    "WebP 최종 이미지를 저장하지 못했습니다.");
            }
        }
        finally
        {
            try
            {
                if (File.Exists(tempPng))
                    File.Delete(tempPng);
            }
            catch
            {
                // 임시파일 정리 실패는 결과 저장을 실패 처리하지 않는다.
            }
        }
    }

    static void SavePlanningDebug(
        string sourcePath,
        IReadOnlyList<RenderUnitPlan> plans,
        string ocrPath,
        string containerPath,
        string unitPath,
        string layoutPath,
        CancellationToken token)
    {
        using var source = Cv2.ImRead(
            sourcePath,
            ImreadModes.Color);

        if (source.Empty())
            return;

        Directory.CreateDirectory(
            Path.GetDirectoryName(ocrPath)!);

        using var ocr = source.Clone();
        using var containers = source.Clone();
        using var units = source.Clone();
        using var layout = source.Clone();

        foreach (var plan in plans)
        {
            token.ThrowIfCancellationRequested();

            foreach (var line in plan.Lines)
            {
                var rect = ClampRect(
                    (int)Math.Floor(line.Source.X),
                    (int)Math.Floor(line.Source.Y),
                    (int)Math.Ceiling(line.Source.W),
                    (int)Math.Ceiling(line.Source.H),
                    source.Cols,
                    source.Rows);

                Cv2.Rectangle(
                    ocr,
                    rect,
                    new Scalar(180, 180, 180),
                    1);

                Cv2.PutText(
                    ocr,
                    line.LineId,
                    new OpenCvSharp.Point(
                        rect.X,
                        Math.Max(12, rect.Y - 2)),
                    HersheyFonts.HersheySimplex,
                    0.35,
                    new Scalar(0, 220, 220),
                    1,
                    LineTypes.AntiAlias);
            }

            Scalar containerColor = plan.Approved
                ? new Scalar(0, 210, 0)
                : new Scalar(0, 0, 230);

            Cv2.Rectangle(
                containers,
                plan.Container.Bounds,
                containerColor,
                2);

            string containerLabel =
                plan.Approved
                    ? $"{plan.ContainerId}:{plan.Container.Mode}"
                    : $"{plan.ContainerId}:{plan.Container.Mode}:KEEP:{CompactPlanReason(plan.Reason)}";

            Cv2.PutText(
                containers,
                containerLabel,
                new OpenCvSharp.Point(
                    plan.Container.Bounds.X,
                    Math.Max(14, plan.Container.Bounds.Y - 3)),
                HersheyFonts.HersheySimplex,
                0.40,
                containerColor,
                1,
                LineTypes.AntiAlias);

            var sourceRect = ClampRect(
                (int)Math.Floor(plan.Region.Source.X),
                (int)Math.Floor(plan.Region.Source.Y),
                (int)Math.Ceiling(plan.Region.Source.W),
                (int)Math.Ceiling(plan.Region.Source.H),
                source.Cols,
                source.Rows);

            Cv2.Rectangle(
                units,
                sourceRect,
                containerColor,
                2);

            Cv2.PutText(
                units,
                $"{plan.UnitId}->{plan.ContainerId}" +
                (plan.Approved ? "" : ":KEEP"),
                new OpenCvSharp.Point(
                    sourceRect.X,
                    Math.Max(14, sourceRect.Y - 3)),
                HersheyFonts.HersheySimplex,
                0.40,
                containerColor,
                1,
                LineTypes.AntiAlias);

            var box = ClampRect(
                (int)Math.Round(plan.Layout.BoxX),
                (int)Math.Round(plan.Layout.BoxY),
                (int)Math.Round(plan.Layout.BoxWidth),
                (int)Math.Round(plan.Layout.BoxHeight),
                source.Cols,
                source.Rows);

            Cv2.Rectangle(
                layout,
                box,
                plan.Layout.Fits
                    ? new Scalar(0, 220, 0)
                    : new Scalar(0, 0, 230),
                2);

            if (plan.Layout.Fits)
            {
                var glyph = ClampRect(
                    (int)Math.Floor(plan.Layout.GlyphX),
                    (int)Math.Floor(plan.Layout.GlyphY),
                    (int)Math.Ceiling(plan.Layout.GlyphWidth),
                    (int)Math.Ceiling(plan.Layout.GlyphHeight),
                    source.Cols,
                    source.Rows);

                Cv2.Rectangle(
                    layout,
                    glyph,
                    new Scalar(220, 120, 0),
                    1);
            }

            Cv2.PutText(
                layout,
                $"{plan.UnitId} {plan.Layout.FontSize:0.0}px" +
                (plan.Layout.Fits ? "" : ":OVERFLOW"),
                new OpenCvSharp.Point(
                    box.X,
                    Math.Max(14, box.Y - 3)),
                HersheyFonts.HersheySimplex,
                0.38,
                plan.Layout.Fits
                    ? new Scalar(220, 120, 0)
                    : new Scalar(0, 0, 230),
                1,
                LineTypes.AntiAlias);
        }

        SaveDebugWebp(ocrPath, ocr);
        SaveDebugWebp(containerPath, containers);
        SaveDebugWebp(unitPath, units);
        SaveDebugWebp(layoutPath, layout);
    }

    static string CompactPlanReason(
        string? reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            return "unknown";

        string value =
            reason
                .Replace(
                    "erase_rejected:",
                    "",
                    StringComparison.Ordinal)
                .Replace(
                    "container_rejected:",
                    "",
                    StringComparison.Ordinal)
                .Replace(
                    ";fallback=keep_original",
                    "",
                    StringComparison.Ordinal);

        return value.Length <= 42
            ? value
            : value[..42];
    }

    static void SaveEraseDebug(
        Mat source,
        Mat mask,
        string path)
    {
        try
        {
            using var debug = source.Clone();

            for (int y = 0; y < mask.Rows; y += 1)
            {
                for (int x = 0; x < mask.Cols; x += 1)
                {
                    if (mask.At<byte>(y, x) == 0)
                        continue;

                    var p = debug.At<Vec3b>(y, x);
                    // BGR: 붉은 오버레이
                    debug.Set(
                        y,
                        x,
                        new Vec3b(
                            (byte)(p.Item0 * 0.35),
                            (byte)(p.Item1 * 0.35),
                            (byte)Math.Min(255, p.Item2 * 0.35 + 165)));
                }
            }

            SaveDebugWebp(path, debug);
        }
        catch
        {
            // debug 실패는 실제 결과를 중단하지 않는다.
        }
    }

    static void WritePlanJson(
        string sourcePath,
        IReadOnlyList<RenderUnitPlan> plans,
        string path)
    {
        try
        {
            var doc = new RenderPlanDocument
            {
                SourceFile = Path.GetFileName(sourcePath),
                Units = plans.Select(x => new RenderPlanDiagnostic
                {
                    UnitId = x.UnitId,
                    ContainerId = x.ContainerId,
                    RegionId = x.Region.Id,
                    LineIds = x.Lines.Select(l => l.LineId).ToList(),
                    Type = x.Region.Type,
                    ContainerDetected = x.Container.Detected,
                    ContainerMode = x.Container.Mode,
                    Approved = x.Approved,
                    Reason = x.Reason,
                    EraseApproved = x.Erase.Approved,
                    EraseReason = x.Erase.Reason,
                    EraseEvidenceLineCount = x.Erase.LineIds.Count,
                    LayoutFits = x.Layout.Fits,
                    LayoutReason = x.Layout.Reason,
                    FontSize = x.Layout.FontSize,
                    BoxX = x.Layout.BoxX,
                    BoxY = x.Layout.BoxY,
                    BoxWidth = x.Layout.BoxWidth,
                    BoxHeight = x.Layout.BoxHeight,
                    GlyphX = x.Layout.GlyphX,
                    GlyphY = x.Layout.GlyphY,
                    GlyphWidth = x.Layout.GlyphWidth,
                    GlyphHeight = x.Layout.GlyphHeight
                }).ToList()
            };

            File.WriteAllText(
                path,
                JsonSerializer.Serialize(
                    doc,
                    new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));
        }
        catch
        {
            // 진단 JSON 실패는 이미지 결과를 실패 처리하지 않는다.
        }
    }

    static double ContainerKeepScore(
        VisionTranslation region,
        BalloonLayout? layout)
    {
        double score = DuplicateKeepScore(region);

        if (layout is not null)
        {
            if (layout.Detected)
                score += 100000;

            if (layout.ShouldRender)
                score += 50000;

            score += layout.LineContainment * 1000;
        }

        return score;
    }

    static bool IsSamePhysicalContainer(
        BalloonLayout a,
        BalloonLayout b)
    {
        double intersection = IntersectionArea(
            a.Bounds,
            b.Bounds);

        double aArea = Math.Max(
            1,
            a.Bounds.Width * (double)a.Bounds.Height);

        double bArea = Math.Max(
            1,
            b.Bounds.Width * (double)b.Bounds.Height);

        double smaller = Math.Min(
            aArea,
            bArea);

        double union = aArea + bArea - intersection;

        double containment = intersection / smaller;
        double iou = union <= 0
            ? 0
            : intersection / union;

        return containment >= 0.88 ||
               iou >= 0.68;
    }

    static bool IsSameCanonicalLine(
        OcrLine a,
        OcrLine b)
    {
        var ar = new OpenCvSharp.Rect2d(
            a.X,
            a.Y,
            Math.Max(1, a.W),
            Math.Max(1, a.H));

        var br = new OpenCvSharp.Rect2d(
            b.X,
            b.Y,
            Math.Max(1, b.W),
            Math.Max(1, b.H));

        double intersection = IntersectionArea(
            ar,
            br);

        double smaller = Math.Max(
            1,
            Math.Min(
                ar.Width * ar.Height,
                br.Width * br.Height));

        if (intersection / smaller < 0.72)
            return false;

        string at = NormalizeDuplicateText(a.Text);
        string bt = NormalizeDuplicateText(b.Text);

        if (at.Length == 0 || bt.Length == 0)
            return true;

        if (at == bt ||
            at.Contains(bt, StringComparison.Ordinal) ||
            bt.Contains(at, StringComparison.Ordinal))
            return true;

        return TokenContainment(at, bt) >= 0.72;
    }

    static double LineMaskCoverage(
        OcrLine line,
        BalloonLayout layout)
    {
        if (!layout.Detected)
            return 0.0;

        int inside = 0;
        int total = 0;

        for (int yi = 0; yi < 3; yi++)
        {
            double ty = 0.20 + yi * 0.30;

            for (int xi = 0; xi < 5; xi++)
            {
                double tx = 0.10 + xi * 0.20;

                int x = (int)Math.Round(
                    line.X + line.W * tx);

                int y = (int)Math.Round(
                    line.Y + line.H * ty);

                total++;

                if (layout.Contains(x, y))
                    inside++;
            }
        }

        return total == 0
            ? 0
            : inside / (double)total;
    }

    static List<VisionTranslation> SuppressDuplicateRegions(
        IReadOnlyList<VisionTranslation> regions,
        out List<VisionTranslation> suppressed)
    {
        suppressed = [];

        if (regions.Count <= 1)
            return regions.ToList();

        var ordered = regions
            .OrderByDescending(DuplicateKeepScore)
            .ThenBy(x => x.Id)
            .ToList();

        var kept = new List<VisionTranslation>(
            ordered.Count);

        foreach (var candidate in ordered)
        {
            bool duplicate = kept.Any(existing =>
                IsDuplicateRenderRegion(
                    existing,
                    candidate));

            if (duplicate)
            {
                suppressed.Add(candidate);
                continue;
            }

            kept.Add(candidate);
        }

        return kept
            .OrderBy(x => x.Id)
            .ToList();
    }

    static double DuplicateKeepScore(
        VisionTranslation region)
    {
        double area = Math.Max(
            1.0,
            region.Source.W * region.Source.H);

        int textLength = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(region.CorrectedText)
                ? region.Source.Text
                : region.CorrectedText).Length;

        return
            region.Source.OriginalRegionCount * 1000.0 +
            textLength * 12.0 +
            Math.Sqrt(area);
    }

    static bool IsDuplicateRenderRegion(
        VisionTranslation a,
        VisionTranslation b)
    {
        if (!string.Equals(
                a.Type,
                b.Type,
                StringComparison.OrdinalIgnoreCase))
            return false;

        var ar = new OpenCvSharp.Rect2d(
            a.Source.X,
            a.Source.Y,
            Math.Max(1, a.Source.W),
            Math.Max(1, a.Source.H));

        var br = new OpenCvSharp.Rect2d(
            b.Source.X,
            b.Source.Y,
            Math.Max(1, b.Source.W),
            Math.Max(1, b.Source.H));

        double intersection = IntersectionArea(
            ar,
            br);

        double smallerArea = Math.Max(
            1.0,
            Math.Min(
                ar.Width * ar.Height,
                br.Width * br.Height));

        double containment =
            intersection / smallerArea;

        if (containment < 0.82)
            return false;

        string at = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(a.CorrectedText)
                ? a.Source.Text
                : a.CorrectedText);

        string bt = NormalizeDuplicateText(
            string.IsNullOrWhiteSpace(b.CorrectedText)
                ? b.Source.Text
                : b.CorrectedText);

        if (at.Length < 4 || bt.Length < 4)
            return false;

        if (at.Contains(bt, StringComparison.Ordinal) ||
            bt.Contains(at, StringComparison.Ordinal))
            return true;

        return containment >= 0.90 &&
               TokenContainment(at, bt) >= 0.78;
    }

    static string NormalizeDuplicateText(
        string text)
    {
        var chars = text
            .ToLowerInvariant()
            .Where(char.IsLetterOrDigit)
            .ToArray();

        return new string(chars);
    }

    static double TokenContainment(
        string a,
        string b)
    {
        static HashSet<string> Tokens(
            string value)
        {
            var result = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < value.Length - 2; i++)
                result.Add(value.Substring(i, 3));

            return result;
        }

        var ta = Tokens(a);
        var tb = Tokens(b);

        if (ta.Count == 0 || tb.Count == 0)
            return 0;

        int common = ta.Count(x => tb.Contains(x));

        return common /
               (double)Math.Min(ta.Count, tb.Count);
    }

    static string NormalizeForAutoLayout(
        string text)
    {
        var normalized = text
            .Replace("[BR]", " ", StringComparison.OrdinalIgnoreCase)
            .Replace('\r', ' ')
            .Replace('\n', ' ')
            .Trim();

        return string.Join(
            " ",
            normalized.Split(
                (char[]?)null,
                StringSplitOptions.RemoveEmptyEntries |
                StringSplitOptions.TrimEntries));
    }

    static Typeface CreateTypeface()
        => new(
            new FontFamily("Malgun Gothic"),
            FontStyles.Normal,
            FontWeights.SemiBold,
            FontStretches.Normal);

    static FormattedText MakeFormattedText(
        string text,
        CultureInfo culture,
        Typeface typeface,
        double fontSize,
        double maxWidth,
        double lineHeightFactor,
        TextAlignment alignment)
        => new(
            text,
            culture,
            FlowDirection.LeftToRight,
            typeface,
            fontSize,
            Brushes.Black,
            1.0)
        {
            TextAlignment = alignment,
            MaxTextWidth = Math.Max(
                12,
                maxWidth),
            LineHeight = fontSize * lineHeightFactor,
            Trimming = TextTrimming.None
        };

    static double ComputePageFontReference(
        IReadOnlyList<VisionTranslation> regions)
    {
        var samples = regions
            .Where(x => x.Render)
            .SelectMany(x => x.Source.Lines)
            .Where(x => x.Confidence >= 0.70f)
            .Select(x => Math.Min(x.W, x.H))
            .Where(x => x >= 10 && x <= 120)
            .OrderBy(x => x)
            .ToArray();

        if (samples.Length == 0)
            return 32;

        int mid = samples.Length / 2;

        return samples.Length % 2 == 1
            ? samples[mid]
            : (samples[mid - 1] + samples[mid]) / 2.0;
    }

    static double ComputeSourceGlyphHint(
        OcrTextBlock block,
        double fallback)
    {
        var samples = block.Lines
            .Where(x => x.Confidence >= 0.60f)
            .Select(x => Math.Min(x.W, x.H))
            .Where(x => x >= 8 && x <= 160)
            .OrderBy(x => x)
            .ToArray();

        if (samples.Length == 0)
            return fallback;

        int mid = samples.Length / 2;

        return samples.Length % 2 == 1
            ? samples[mid]
            : (samples[mid - 1] + samples[mid]) / 2.0;
    }

    static void SaveSourceAsOutput(
        string sourcePath,
        string outputPath)
    {
        using var source = Cv2.ImRead(
            sourcePath,
            ImreadModes.Color);

        if (source.Empty() ||
            !SaveLosslessWebp(
                outputPath,
                source))
        {
            throw new InvalidOperationException(
                "원문 보존 이미지를 저장하지 못했습니다.");
        }
    }

    static bool SaveLosslessWebp(
        string path,
        Mat image)
        => Cv2.ImWrite(
            path,
            image,
            new[]
            {
                new ImageEncodingParam(
                    ImwriteFlags.WebPQuality,
                    FinalWebpLosslessQuality)
            });

    static bool SaveDebugWebp(
        string path,
        Mat image)
        => Cv2.ImWrite(
            path,
            image,
            new[]
            {
                new ImageEncodingParam(
                    ImwriteFlags.WebPQuality,
                    DebugWebpQuality)
            });

    static void SaveDebugCopy(
        string sourcePath,
        string outputPath)
    {
        using var image = Cv2.ImRead(
            sourcePath,
            ImreadModes.Color);

        if (image.Empty())
            return;

        SaveDebugWebp(
            outputPath,
            image);
    }

    static BitmapSource LoadBitmap(
        string path)
    {
        using var input = File.OpenRead(path);

        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = input;
        bitmap.EndInit();
        bitmap.Freeze();

        return bitmap;
    }

    static CvRect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left = Math.Clamp(
            x,
            0,
            Math.Max(0, imageWidth - 1));

        int top = Math.Clamp(
            y,
            0,
            Math.Max(0, imageHeight - 1));

        int right = Math.Clamp(
            x + Math.Max(1, width),
            left + 1,
            imageWidth);

        int bottom = Math.Clamp(
            y + Math.Max(1, height),
            top + 1,
            imageHeight);

        return new CvRect(
            left,
            top,
            Math.Max(1, right - left),
            Math.Max(1, bottom - top));
    }

    static double IntersectionArea(
        CvRect a,
        CvRect b)
    {
        int left = Math.Max(a.Left, b.Left);
        int top = Math.Max(a.Top, b.Top);
        int right = Math.Min(a.Right, b.Right);
        int bottom = Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               (double)Math.Max(0, bottom - top);
    }

    static double IntersectionArea(
        OpenCvSharp.Rect2d a,
        OpenCvSharp.Rect2d b)
    {
        double left = Math.Max(a.Left, b.Left);
        double top = Math.Max(a.Top, b.Top);
        double right = Math.Min(a.Right, b.Right);
        double bottom = Math.Min(a.Bottom, b.Bottom);

        return Math.Max(0, right - left) *
               Math.Max(0, bottom - top);
    }

    static Vec3b MedianColor(
        IReadOnlyList<Vec3b> samples)
    {
        if (samples.Count == 0)
            return new Vec3b(255, 255, 255);

        var b = samples
            .Select(x => (int)x.Item0)
            .OrderBy(x => x)
            .ToArray();

        var g = samples
            .Select(x => (int)x.Item1)
            .OrderBy(x => x)
            .ToArray();

        var r = samples
            .Select(x => (int)x.Item2)
            .OrderBy(x => x)
            .ToArray();

        int mid = samples.Count / 2;

        return new Vec3b(
            (byte)b[mid],
            (byte)g[mid],
            (byte)r[mid]);
    }

    static double ColorDistance(
        Vec3b a,
        Vec3b b)
    {
        double db = a.Item0 - b.Item0;
        double dg = a.Item1 - b.Item1;
        double dr = a.Item2 - b.Item2;

        return Math.Sqrt(
            db * db +
            dg * dg +
            dr * dr);
    }

    static double Luminance(
        Vec3b p)
        => p.Item2 * 0.299 +
           p.Item1 * 0.587 +
           p.Item0 * 0.114;

    static double Percentile(
        IReadOnlyList<double> sorted,
        double percentile)
    {
        if (sorted.Count == 0)
            return 0;

        double p = Math.Clamp(
            percentile,
            0,
            1);

        int index = (int)Math.Round(
            (sorted.Count - 1) * p);

        return sorted[
            Math.Clamp(
                index,
                0,
                sorted.Count - 1)];
    }
}


public static class RenderSafetyPolicy
{
    public static bool IsSuspiciousVisionExpansion(
        VisionTranslation region)
    {
        if (region.Source.Lines.Count == 0 ||
            region.Source.Lines.Count > 2)
            return false;

        int sourceLength = region.Source.Text.Count(
            char.IsLetterOrDigit);

        if (sourceLength == 0 ||
            sourceLength > 4)
            return false;

        double confidence = region.Source.Lines.Average(
            x => x.Confidence);

        if (confidence >= 0.82)
            return false;

        int correctedLength = region.CorrectedText.Count(
            char.IsLetterOrDigit);

        return correctedLength >= 8 &&
               correctedLength >= sourceLength * 2.5;
    }
}
