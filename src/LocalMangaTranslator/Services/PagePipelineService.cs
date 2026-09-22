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

            await laya.WriteShadowAuditAsync(
                sourcePath,
                outputDirectory,
                pageAnalysisResult,
                ocrStage,
                visionReviewedBeforeRecovery,
                translated,
                options,
                layaProgress,
                token);
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

                if (!ShouldRecoverDetectorOwnedRenderableUnit(
                        region,
                        regions))
                {
                    return region;
                }

                return region with
                {
                    Render = true,
                    Source =
                        detectorTextRegion is null
                            ? region.Source
                            : region.Source with
                            {
                                RegionTextRegion =
                                    detectorTextRegion
                            }
                };
            })
            .OrderBy(x =>
                x.Id)
            .ToList();
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

