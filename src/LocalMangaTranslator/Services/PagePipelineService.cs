using System.Text.Json;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.PipelineV2.Erase;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Coordinates one page through the major pipeline stages.
/// UI code should only provide models, output location and progress handling.
/// This service intentionally keeps stage boundaries explicit so future
/// container-first detection, container OCR and post-render validation can be
/// inserted without coupling them to MainWindow.
/// </summary>
public sealed class PagePipelineService
{
    readonly PageAnalysisService pageAnalysis;
    readonly OcrPipelineService ocrPipeline;
    readonly VisionTranslationService vision;
    readonly TranslationRefinementService translationRefiner;
    readonly RenderPipelineService renderer;
    readonly FinalAuditService finalAudit = new();
    readonly LayaDecisionService laya;

    public PagePipelineService(
        PageAnalysisService pageAnalysis,
        OcrPipelineService ocrPipeline,
        VisionTranslationService vision,
        TranslationRefinementService translationRefiner,
        RenderPipelineService renderer,
        LayaDecisionService? laya = null)
    {
        this.pageAnalysis = pageAnalysis;
        this.ocrPipeline = ocrPipeline;
        this.vision = vision;
        this.translationRefiner = translationRefiner;
        this.renderer = renderer;
        this.laya = laya ?? new LayaDecisionService();
    }

    public async Task<PagePipelineResult> ProcessAsync(
        string sourcePath,
        string outputDirectory,
        ModelProfile reviewModel,
        ModelProfile translationModel,
        PipelineOptions options,
        IProgress<PipelineProgress>? progress = null,
        CancellationToken token = default)
    {
        OutputDirectoryLayout.Ensure(outputDirectory);

        progress?.Report(new PipelineProgress(
            PipelineStageKind.PageAnalysis,
            options.RegionAnalysis == RegionAnalysisMode.HybridRtdetr
                ? "페이지 구조 분석 · RT-DETR + 기존 컨테이너 검출"
                : "페이지 구조 분석 · 기존 컨테이너 검출"));

        var pageAnalysisResult =
            await pageAnalysis.AnalyzeAsync(
                sourcePath,
                options,
                token);

        PipelineDebugWriter.SavePageAnalysisDiagnostics(
            sourcePath,
            outputDirectory,
            pageAnalysisResult);

        // Freeze detector geometry before OCR/translation can reinterpret it.
        // V2 erase itself runs later, after translation, so untranslated
        // TextBubble targets are never blanked accidentally.
        V2DetectionSnapshot? v2Snapshot =
            pageAnalysisResult.ExternalDetectorUsed
                ? V2DetectionSnapshot.Create(
                    pageAnalysisResult)
                : null;

        if (v2Snapshot is not null)
        {
            progress?.Report(new PipelineProgress(
                PipelineStageKind.PageAnalysis,
                $"Pipeline V2 · RT-DETR geometry 고정 · TextBubble {v2Snapshot.TextTargets.Count}개"));
        }

        progress?.Report(new PipelineProgress(
            PipelineStageKind.PageAnalysis,
            $"페이지 구조 분석 완료 · {pageAnalysisResult.Mode} · " +
            $"영역 {pageAnalysisResult.Regions.Count}개 / 컨테이너 {pageAnalysisResult.ContainerCandidates.Count}개"));

        progress?.Report(new PipelineProgress(
            PipelineStageKind.OcrObservation,
            "OCR 관측 / 병합 시작"));

        var ocrStage =
            await ocrPipeline.AnalyzeAsync(
                sourcePath,
                pageAnalysisResult,
                options,
                token,
                progress);

        PipelineDebugWriter.SavePreVisionDiagnostics(
            sourcePath,
            outputDirectory,
            ocrStage);

        progress?.Report(new PipelineProgress(
            PipelineStageKind.OcrObservation,
            $"페이지 컨테이너 후보 {ocrStage.PageCandidates.Count}개 · " +
            $"{OcrPipelineService.FormatPassSummary(ocrStage.Observations)} → 병합 {ocrStage.MergedLines.Count}줄"));

        if (ocrStage.UnitBuild.Units.Count == 0)
        {
            return new PagePipelineResult(
                false,
                ocrStage,
                [],
                [],
                null,
                null);
        }

        var unitBuild =
            ocrStage.UnitBuild;

        var blocks =
            unitBuild.Units.ToList();

        progress?.Report(new PipelineProgress(
            PipelineStageKind.OcrUnitFormation,
            $"OCR unit 구성 · {ocrStage.MergedLines.Count}줄 → 예비 {ocrStage.PreliminaryBlocks.Count}블록 → " +
            $"컨테이너 {unitBuild.ContainerCount}개 / 배정 {unitBuild.AssignedLineCount}줄 / " +
            $"고아 {unitBuild.OrphanGroupCount}블록 → Vision 입력 {blocks.Count} unit"));

        progress?.Report(new PipelineProgress(
            PipelineStageKind.VisionReview,
            $"Vision OCR 검수 시작 · {reviewModel.ModelTag}"));

        var visionProgress =
            new Progress<string>(message =>
                progress?.Report(new PipelineProgress(
                    PipelineStageKind.VisionReview,
                    message)));

        var reviewed =
            await vision.ReviewAndTranslateAsync(
                sourcePath,
                blocks,
                reviewModel,
                visionProgress,
                token);

        // 0036 shadow track compares Laya against the raw Vision decision
        // before any Classic detector-owned recovery mutates Render.
        var visionReviewedBeforeRecovery =
            reviewed.ToList();

        int corrected =
            reviewed.Count(x =>
                !string.Equals(
                    x.Source.Text.Trim(),
                    x.CorrectedText.Trim(),
                    StringComparison.Ordinal));

        int skipped =
            reviewed.Count(x => !x.Render);

        var detectorRecovered =
            RecoverDetectorOwnedRenderableUnits(
                reviewed,
                pageAnalysisResult.Regions);

        int detectorRecoveredCount =
            detectorRecovered.Count(x =>
                x.Render) -
            reviewed.Count(x =>
                x.Render);

        reviewed =
            detectorRecovered;

        progress?.Report(new PipelineProgress(
            PipelineStageKind.VisionReview,
            $"Vision 검수 완료 · {reviewed.Count}개 블록 · OCR 교정 {corrected}개 · " +
            $"조판 제외 {skipped}개 · detector-owned 대사 복구 {detectorRecoveredCount}개"));

        progress?.Report(new PipelineProgress(
            PipelineStageKind.Translation,
            $"최종 번역 시작 · {translationModel.ModelTag}"));

        var translationProgress =
            new Progress<string>(message =>
                progress?.Report(new PipelineProgress(
                    PipelineStageKind.Translation,
                    message)));

        var translated =
            await translationRefiner.TranslateAsync(
                reviewed,
                translationModel,
                translationProgress,
                token);

        progress?.Report(new PipelineProgress(
            PipelineStageKind.Translation,
            $"최종 번역 완료 · {translated.Count(x => x.Render)}개 조판 대상"));

        V2EraseSelection? v2Selection =
            null;

        if (v2Snapshot is not null)
        {
            v2Selection =
                V2EraseSelector.Select(
                    v2Snapshot,
                    translated);

            progress?.Report(new PipelineProgress(
                PipelineStageKind.Render,
                $"Pipeline V2 · TextBubble erase / parent Bubble layout 연결 " +
                $"{v2Selection.Bindings.Count} Unit · TextBubble {v2Selection.TextRegionIds.Count}개"));
        }

        if (options.DecisionTrack ==
                PipelineDecisionTrack.LayaExperimental &&
            options.LayaMode !=
                LayaDecisionMode.Off)
        {
            var layaProgress =
                new Progress<string>(message =>
                    progress?.Report(
                        new PipelineProgress(
                            PipelineStageKind.VisionReview,
                            message)));

            var layaAudit =
                await laya.WriteShadowAuditAsync(
                    sourcePath,
                    outputDirectory,
                    pageAnalysisResult,
                    ocrStage,
                    visionReviewedBeforeRecovery,
                    reviewed,
                    translated,
                    v2Selection,
                    options,
                    layaProgress,
                    token);

            if (layaAudit.DetectorRetryRequests.Count >
                    0 &&
                pageAnalysisResult.ExternalDetectorUsed)
            {
                var retryProgress =
                    new Progress<string>(message =>
                        progress?.Report(
                            new PipelineProgress(
                                PipelineStageKind.PageAnalysis,
                                message)));

                var retryAudit =
                    await pageAnalysis
                        .RunDetectorRetryAuditAsync(
                            sourcePath,
                            outputDirectory,
                            layaAudit.DetectorRetryRequests,
                            pageAnalysisResult.Regions,
                            options.LayaMode,
                            retryProgress,
                            token);

                if (retryAudit.AdvisoryCandidates.Count >
                        0 &&
                    v2Snapshot is not null &&
                    v2Selection is not null)
                {
                    (
                        translated,
                        v2Snapshot,
                        v2Selection) =
                        ApplyDetectorRetryAdvisory(
                            translated,
                            v2Snapshot,
                            v2Selection,
                            retryAudit.AdvisoryCandidates);

                    progress?.Report(
                        new PipelineProgress(
                            PipelineStageKind.PageAnalysis,
                            $"Laya Advisory · 신규 TextBubble {retryAudit.AdvisoryCandidates.Count}개를 " +
                            "해당 UNBOUND unit에만 제한 적용"));
                }
            }

            if (layaAudit.RenderRecoveryRequests.Count >
                0)
            {
                await WriteShadowRenderRecoveryAuditAsync(
                    sourcePath,
                    outputDirectory,
                    translationModel,
                    v2Snapshot,
                    layaAudit.RenderRecoveryRequests,
                    progress,
                    token);
            }
        }

        var document =
            new VisionTranslationDocument
            {
                SourceFile = Path.GetFileName(sourcePath),
                Model =
                    $"review={reviewModel.ModelTag}; translation={translationModel.ModelTag}",
                Regions = translated
            };

        var jsonPath =
            Path.Combine(
                OutputDirectoryLayout.Json(outputDirectory),
                Path.GetFileNameWithoutExtension(sourcePath) +
                ".translation.json");

        await File.WriteAllTextAsync(
            jsonPath,
            JsonSerializer.Serialize(
                document,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }),
            token);

        progress?.Report(new PipelineProgress(
            PipelineStageKind.Translation,
            $"번역 JSON 저장: json/{Path.GetFileName(jsonPath)}"));

        var imagePath =
            Path.Combine(
                outputDirectory,
                Path.GetFileNameWithoutExtension(sourcePath) +
                ".translated.webp");

        progress?.Report(new PipelineProgress(
            PipelineStageKind.Render,
            v2Selection is not null
                ? "Pipeline V2 · layout 사전검증 → erase 검수 → 원자적 조판"
                : "RenderPlan / legacy erase / layout / 조판 시작"));

        var renderProgress =
            new Progress<string>(message =>
                progress?.Report(new PipelineProgress(
                    PipelineStageKind.Render,
                    message)));

        await renderer.RenderAsync(
            sourcePath,
            translated,
            imagePath,
            renderProgress,
            token,
            v2Snapshot:
                v2Snapshot,
            v2Selection:
                v2Selection);

        // Final image delivery is complete at this point. Audit is diagnostic
        // only and can never invalidate or remove the finished output.
        if (options.EnableFinalAudit)
        {
            progress?.Report(new PipelineProgress(
                PipelineStageKind.FinalAudit,
                "완료 이미지 검토 로그 생성"));

            try
            {
                await finalAudit.GenerateAsync(
                    sourcePath,
                    imagePath,
                    outputDirectory,
                    ocrStage,
                    translated,
                    token);

                progress?.Report(new PipelineProgress(
                    PipelineStageKind.FinalAudit,
                    $"완료검토로그/{Path.GetFileNameWithoutExtension(sourcePath)}.final_compare.webp"));
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                progress?.Report(new PipelineProgress(
                    PipelineStageKind.FinalAudit,
                    $"완료 검토 로그 생성 실패 · 결과 이미지는 정상 저장됨 · {ex.Message}"));
            }
        }

        progress?.Report(new PipelineProgress(
            PipelineStageKind.Completed,
            $"완료 이미지: {Path.GetFileName(imagePath)}"));

        return new PagePipelineResult(
            true,
            ocrStage,
            reviewed,
            translated,
            jsonPath,
            imagePath);
    }

    static (
        List<VisionTranslation> Translated,
        V2DetectionSnapshot Snapshot,
        V2EraseSelection Selection)
        ApplyDetectorRetryAdvisory(
            IReadOnlyList<VisionTranslation> translated,
            V2DetectionSnapshot snapshot,
            V2EraseSelection existingSelection,
            IReadOnlyList<LayaDetectorAdvisoryCandidate> candidates)
    {
        var updated =
            translated.ToList();

        var rawRegions =
            snapshot.RawRegions.ToList();

        var currentSnapshot =
            snapshot;

        var targetIds =
            new HashSet<string>(
                existingSelection.TextRegionIds,
                StringComparer.Ordinal);

        var translationIds =
            new HashSet<int>(
                existingSelection.TranslationRegionIds);

        var bindings =
            existingSelection.Bindings
                .ToDictionary(
                    x =>
                        x.Key,
                    x =>
                        x.Value);

        var preservationReasons =
            existingSelection.PreservationReasons
                .ToDictionary(
                    x =>
                        x.Key,
                    x =>
                        x.Value);

        var preservationTargetReasons =
            existingSelection.PreservationTargetReasons
                .ToDictionary(
                    x =>
                        x.Key,
                    x =>
                        x.Value,
                    StringComparer.Ordinal);

        foreach (var candidate in
                 candidates)
        {
            int index =
                updated.FindIndex(x =>
                    x.Id ==
                    candidate.TranslationRegionId);

            if (index < 0)
                continue;

            var current =
                updated[index];

            if (!current.Render ||
                string.IsNullOrWhiteSpace(
                    current.Translation))
            {
                continue;
            }

            var linked =
                current with
                {
                    Source =
                        current.Source with
                        {
                            RegionId =
                                candidate.BubbleRegion.RegionId,
                            RegionTextRegion =
                                candidate.TextRegion
                        }
                };

            var tentativeRaw =
                rawRegions.ToList();

            if (!tentativeRaw.Any(x =>
                    string.Equals(
                        x.RegionId,
                        candidate.BubbleRegion.RegionId,
                        StringComparison.Ordinal)))
            {
                tentativeRaw.Add(
                    candidate.BubbleRegion);
            }

            if (!tentativeRaw.Any(x =>
                    string.Equals(
                        x.RegionId,
                        candidate.TextRegion.RegionId,
                        StringComparison.Ordinal)))
            {
                tentativeRaw.Add(
                    candidate.TextRegion);
            }

            var tentativeSnapshot =
                V2DetectionSnapshot.Create(
                    new PageAnalysisResult(
                        tentativeRaw,
                        [],
                        snapshot.SourceMode +
                        "+laya_advisory",
                        true,
                        "detector_retry_advisory"));

            var one =
                V2EraseSelector.Select(
                    tentativeSnapshot,
                    [linked]);

            if (!one.Bindings.TryGetValue(
                    linked.Id,
                    out var binding) ||
                !binding.TextRegionIds.Contains(
                    candidate.TextRegion.RegionId,
                    StringComparer.Ordinal))
            {
                continue;
            }

            // Only now commit this advisory target. A failed candidate never
            // enters the immutable target set, so it cannot create a new
            // unresolved coverage target.
            updated[index] =
                linked;

            rawRegions =
                tentativeRaw;

            currentSnapshot =
                tentativeSnapshot;

            foreach (string id in
                     one.TextRegionIds)
            {
                targetIds.Add(
                    id);
            }

            translationIds.Add(
                linked.Id);

            bindings[linked.Id] =
                binding;

            preservationReasons.Remove(
                linked.Id);

            foreach (string id in
                     binding.TextRegionIds)
            {
                preservationTargetReasons.Remove(
                    id);
            }
        }

        return (
            updated,
            currentSnapshot,
            new V2EraseSelection(
                targetIds,
                translationIds,
                bindings)
            {
                PreservationReasons =
                    preservationReasons,
                PreservationTargetReasons =
                    preservationTargetReasons
            });
    }

    async Task WriteShadowRenderRecoveryAuditAsync(
        string sourcePath,
        string outputDirectory,
        ModelProfile translationModel,
        V2DetectionSnapshot? snapshot,
        IReadOnlyList<LayaRenderRecoveryRequest> requests,
        IProgress<PipelineProgress>? progress,
        CancellationToken token)
    {
        var items =
            new List<object>();

        foreach (var request in requests)
        {
            token.ThrowIfCancellationRequested();

            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.Translation,
                    $"Laya render-recovery Shadow 번역 · id={request.Region.Id}"));

            var rearmed =
                request.Region with
                {
                    Render =
                        true
                };

            var shadowProgress =
                new Progress<string>(message =>
                    progress?.Report(
                        new PipelineProgress(
                            PipelineStageKind.Translation,
                            $"Laya render-recovery Shadow · {message}")));

            var shadowTranslated =
                await translationRefiner
                    .TranslateAsync(
                        [rearmed],
                        translationModel,
                        shadowProgress,
                        token);

            var candidate =
                shadowTranslated
                    .FirstOrDefault();

            V2EraseSelection? shadowSelection =
                snapshot is not null &&
                candidate is not null
                    ? V2EraseSelector.Select(
                        snapshot,
                        [candidate])
                    : null;

            V2RegionBinding? binding =
                null;

            if (candidate is not null &&
                shadowSelection is not null)
            {
                shadowSelection.Bindings.TryGetValue(
                    candidate.Id,
                    out binding);
            }

            items.Add(
                new
                {
                    TranslationRegionId =
                        request.Region.Id,
                    request.Confidence,
                    SourceText =
                        request.Region.Source.Text,
                    CorrectedText =
                        request.Region.CorrectedText,
                    OriginalFinalRender =
                        request.Region.Render,
                    ShadowCandidate =
                        candidate is null
                            ? null
                            : new
                            {
                                candidate.Render,
                                candidate.Translation,
                                candidate.Type
                            },
                    V2Binding =
                        binding is null
                            ? null
                            : new
                            {
                                binding.TextRegionIds,
                                binding.BubbleRegionId,
                                TextBounds =
                                    new
                                    {
                                        binding.TextBounds.X,
                                        binding.TextBounds.Y,
                                        binding.TextBounds.Width,
                                        binding.TextBounds.Height
                                    },
                                LayoutBounds =
                                    new
                                    {
                                        binding.LayoutBounds.X,
                                        binding.LayoutBounds.Y,
                                        binding.LayoutBounds.Width,
                                        binding.LayoutBounds.Height
                                    },
                                binding.LayoutMode
                            },
                    WouldBecomeRenderable =
                        candidate is not null &&
                        candidate.Render &&
                        !string.IsNullOrWhiteSpace(
                            candidate.Translation) &&
                        binding is not null,
                    MutationApplied =
                        false
                });
        }

        string page =
            Path.GetFileNameWithoutExtension(
                sourcePath);

        string path =
            Path.Combine(
                OutputDirectoryLayout.Debug(
                    outputDirectory),
                page +
                ".laya_render_recovery.json");

        await File.WriteAllTextAsync(
            path,
            JsonSerializer.Serialize(
                new
                {
                    Schema =
                        "pipeline-v2-laya-render-recovery-shadow-v1",
                    SourceFile =
                        Path.GetFileName(
                            sourcePath),
                    Mode =
                        "Shadow",
                    MutationApplied =
                        false,
                    Requests =
                        requests.Count,
                    Items =
                        items
                },
                new JsonSerializerOptions
                {
                    WriteIndented =
                        true
                }),
            token);
    }

    static List<VisionTranslation> RecoverDetectorOwnedRenderableUnits(
        IReadOnlyList<VisionTranslation> reviewed,
        IReadOnlyList<PageRegion> regions)
    {
        return reviewed
            .Select(region =>
            {
                var detectorTextRegion =
                    ResolveDetectorOwnedTextRegion(
                        region.Source,
                        regions);

                bool promoteSecondary =
                    ShouldPromoteSecondaryOcrForDetectorOwnedSpeech(
                        region,
                        regions);

                bool recoverBubbleSign =
                    ShouldRecoverDetectorOwnedSignAsCaption(
                        region,
                        regions);

                if (!ShouldRecoverDetectorOwnedRenderableUnit(
                        region,
                        regions) &&
                    !promoteSecondary &&
                    !recoverBubbleSign)
                {
                    return region;
                }

                var recoveredSource =
                    detectorTextRegion is null
                        ? region.Source
                        : region.Source with
                        {
                            RegionTextRegion =
                                detectorTextRegion
                        };

                // 0049: when primary OCR is only a low-confidence fragment but
                // an independent Baberu pass reads a substantially longer
                // sentence inside the same immutable speech TextBubble, promote
                // only the semantic text. Detector geometry and erase ownership
                // remain unchanged.
                if (promoteSecondary)
                {
                    return region with
                    {
                        Source =
                            recoveredSource,
                        CorrectedText =
                            region.Source.SecondaryOcrText!.Trim(),
                        Translation = "",
                        Type = "dialogue",
                        Render = true
                    };
                }

                // 0049: a high-confidence primary/secondary agreement inside a
                // detector-owned speech TextBubble overrides Vision's generic
                // sign label. Treat it as a renderable device/narrative caption
                // without broadening the global sign policy.
                if (recoverBubbleSign)
                {
                    return region with
                    {
                        Source =
                            recoveredSource,
                        Type = "caption",
                        Render = true
                    };
                }

                return region with
                {
                    Render = true,
                    Source =
                        recoveredSource
                };
            })
            .OrderBy(x =>
                x.Id)
            .ToList();
    }

    public static bool ShouldPromoteSecondaryOcrForDetectorOwnedSpeech(
        VisionTranslation region,
        IReadOnlyList<PageRegion>? regions = null)
    {
        if (region.Render ||
            !string.Equals(
                region.Type,
                "other",
                StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var textRegion =
            ResolveDetectorOwnedTextRegion(
                region.Source,
                regions);

        if (textRegion is null ||
            textRegion.Kind !=
                PageRegionKind.TextBubble ||
            region.Source.RegionContainer?.Kind !=
                ContainerCandidateKind.Speech)
        {
            return false;
        }

        if (!string.Equals(
                region.Source.SecondaryOcrAgreement,
                "disagree",
                StringComparison.OrdinalIgnoreCase) ||
            string.IsNullOrWhiteSpace(
                region.Source.SecondaryOcrText))
        {
            return false;
        }

        int primaryLength =
            region.Source.Text.Count(
                char.IsLetterOrDigit);

        int correctedLength =
            (string.IsNullOrWhiteSpace(
                 region.CorrectedText)
                ? region.Source.Text
                : region.CorrectedText)
            .Count(
                char.IsLetterOrDigit);

        int secondaryLength =
            region.Source.SecondaryOcrText.Count(
                char.IsLetterOrDigit);

        if (primaryLength is < 1 or > 6 ||
            correctedLength >
                Math.Max(
                    6,
                    primaryLength + 3) ||
            secondaryLength <
                Math.Max(
                    10,
                    primaryLength * 3))
        {
            return false;
        }

        double averageConfidence =
            region.Source.Lines.Count == 0
                ? 0
                : region.Source.Lines.Average(x =>
                    x.Confidence);

        if (region.Source.Lines.Count == 0 ||
            averageConfidence >= 0.55)
        {
            return false;
        }

        int secondaryWords =
            region.Source.SecondaryOcrText
                .Split(
                    (char[]?)null,
                    StringSplitOptions.RemoveEmptyEntries |
                    StringSplitOptions.TrimEntries)
                .Count(x =>
                    x.Any(
                        char.IsLetterOrDigit));

        return secondaryWords >= 3;
    }

    public static bool ShouldRecoverDetectorOwnedSignAsCaption(
        VisionTranslation region,
        IReadOnlyList<PageRegion>? regions = null)
    {
        if (region.Render ||
            !string.Equals(
                region.Type,
                "sign",
                StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var textRegion =
            ResolveDetectorOwnedTextRegion(
                region.Source,
                regions);

        if (textRegion is null ||
            textRegion.Kind !=
                PageRegionKind.TextBubble ||
            region.Source.RegionContainer?.Kind !=
                ContainerCandidateKind.Speech)
        {
            return false;
        }

        if (!string.Equals(
                region.Source.SecondaryOcrAgreement,
                "agree",
                StringComparison.OrdinalIgnoreCase) ||
            string.IsNullOrWhiteSpace(
                region.Source.SecondaryOcrText))
        {
            return false;
        }

        int primaryLength =
            region.Source.Text.Count(
                char.IsLetterOrDigit);

        int secondaryLength =
            region.Source.SecondaryOcrText.Count(
                char.IsLetterOrDigit);

        if (primaryLength < 8 ||
            secondaryLength < 8)
        {
            return false;
        }

        double averageConfidence =
            region.Source.Lines.Count == 0
                ? 0
                : region.Source.Lines.Average(x =>
                    x.Confidence);

        return averageConfidence >= 0.90;
    }

    public static bool ShouldRecoverDetectorOwnedRenderableUnit(
        VisionTranslation region,
        IReadOnlyList<PageRegion>? regions = null)
    {
        if (region.Render)
            return false;

        var textRegion =
            ResolveDetectorOwnedTextRegion(
                region.Source,
                regions);

        if (textRegion is null ||
            textRegion.Kind !=
                PageRegionKind.TextBubble ||
            region.Source.RegionContainer is null)
        {
            return false;
        }

        bool renderableType =
            region.Type is
                "dialogue" or
                "thought" or
                "caption";

        if (!renderableType)
            return false;

        string corrected =
            string.IsNullOrWhiteSpace(
                region.CorrectedText)
                ? region.Source.Text
                : region.CorrectedText;

        int meaningful =
            corrected.Count(
                char.IsLetterOrDigit);

        if (meaningful < 2)
            return false;

        bool secondaryEvidence =
            !string.IsNullOrWhiteSpace(
                region.Source.SecondaryOcrText) &&
            region.Source.SecondaryOcrText.Count(
                char.IsLetterOrDigit) >= 2;

        double averageConfidence =
            region.Source.Lines.Count == 0
                ? 0
                : region.Source.Lines.Average(x =>
                    x.Confidence);

        bool strongPrimary =
            region.Source.Lines.Count > 0 &&
            averageConfidence >= 0.80;

        // 0035: a detector-owned Bubble with one unambiguous TextBubble child
        // is sufficient structural evidence even when the runtime
        // RegionTextRegion link was lost earlier.  The unit is only re-armed
        // for final translation; erase/layout still need the immutable target.
        return secondaryEvidence ||
               strongPrimary;
    }

    static PageRegion? ResolveDetectorOwnedTextRegion(
        OcrTextBlock source,
        IReadOnlyList<PageRegion>? regions)
    {
        if (source.RegionTextRegion is { } direct &&
            direct.Kind ==
                PageRegionKind.TextBubble)
        {
            return direct;
        }

        if (regions is null ||
            string.IsNullOrWhiteSpace(
                source.RegionId))
        {
            return null;
        }

        var parentBubble =
            regions.FirstOrDefault(x =>
                x.Kind ==
                    PageRegionKind.Bubble &&
                string.Equals(
                    x.RegionId,
                    source.RegionId,
                    StringComparison.Ordinal));

        if (parentBubble is null)
            return null;

        var candidates =
            regions
                .Where(x =>
                    x.Kind ==
                        PageRegionKind.TextBubble)
                .Select(text => new
                {
                    Text = text,
                    Parent =
                        FindParentBubble(
                            text,
                            regions)
                })
                .Where(x =>
                    x.Parent is not null &&
                    string.Equals(
                        x.Parent.RegionId,
                        parentBubble.RegionId,
                        StringComparison.Ordinal))
                .Select(x =>
                    x.Text)
                .ToList();

        return candidates.Count == 1
            ? candidates[0]
            : null;
    }

    static PageRegion? FindParentBubble(
        PageRegion textRegion,
        IReadOnlyList<PageRegion> regions)
    {
        return regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.Bubble)
            .Select(x => new
            {
                Region = x,
                Coverage =
                    RegionCoverage(
                        textRegion.Bounds,
                        x.Bounds),
                CenterInside =
                    ContainsRegionCenter(
                        x.Bounds,
                        textRegion.Bounds)
            })
            .Where(x =>
                x.CenterInside ||
                x.Coverage >= 0.45)
            .OrderByDescending(x =>
                x.CenterInside)
            .ThenByDescending(x =>
                x.Coverage)
            .ThenByDescending(x =>
                x.Region.Score)
            .Select(x =>
                x.Region)
            .FirstOrDefault();
    }

    static double RegionCoverage(
        OpenCvSharp.Rect inner,
        OpenCvSharp.Rect outer)
    {
        int left =
            Math.Max(
                inner.Left,
                outer.Left);

        int top =
            Math.Max(
                inner.Top,
                outer.Top);

        int right =
            Math.Min(
                inner.Right,
                outer.Right);

        int bottom =
            Math.Min(
                inner.Bottom,
                outer.Bottom);

        double intersection =
            Math.Max(
                0,
                right - left) *
            (double)Math.Max(
                0,
                bottom - top);

        double area =
            Math.Max(
                1.0,
                inner.Width *
                (double)inner.Height);

        return intersection /
               area;
    }

    static bool ContainsRegionCenter(
        OpenCvSharp.Rect container,
        OpenCvSharp.Rect subject)
    {
        double cx =
            subject.X +
            subject.Width / 2.0;

        double cy =
            subject.Y +
            subject.Height / 2.0;

        return cx >= container.Left &&
               cx <= container.Right &&
               cy >= container.Top &&
               cy <= container.Bottom;
    }

}

