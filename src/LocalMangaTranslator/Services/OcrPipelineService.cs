using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

/// <summary>
/// OCR branch of Architecture 2.0.
/// Page/layout analysis is supplied from the independent PageAnalysisService.
/// This stage only acquires/fuses text observations and forms translation units.
/// </summary>
public sealed class OcrPipelineService
{
    readonly OcrEngine ocr;
    readonly OcrBlockGrouper blockGrouper = new();
    readonly OcrContainerUnitBuilder unitBuilder = new();

    public OcrPipelineService(
        OcrEngine ocr)
    {
        this.ocr = ocr;
    }

    public async Task<OcrStageResult> AnalyzeAsync(
        string sourcePath,
        PageAnalysisResult pageAnalysis,
        PipelineOptions options,
        CancellationToken token = default,
        IProgress<PipelineProgress>? progress = null)
    {
        var pageCandidates =
            pageAnalysis.ContainerCandidates.ToList();

        var candidateDecisions =
            pageCandidates
                .Select(
                    ContainerOcrValidator.ValidateCandidate)
                .ToList();

        var eligibleIds =
            candidateDecisions
                .Where(x => x.Eligible)
                .Select(x => x.CandidateId)
                .ToHashSet();

        var eligible =
            pageCandidates
                .Where(
                    x =>
                        eligibleIds.Contains(
                            x.CandidateId))
                .ToList();

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.ContainerValidation,
                $"컨테이너 후보 검증 · {pageCandidates.Count}개 중 OCR 대상 {eligible.Count}개"));

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.OcrObservation,
                "전체 1배·2배 / 집중 3배 OCR"));

        var observationBatch =
            await ocr.RecognizeDetailedAsync(
                sourcePath,
                token);

        RegionOcrBatch regionBatch =
            new(
                [],
                []);

        IReadOnlyList<RegionLineDecision> regionLineDecisions =
            [];

        if (options.EnableTargetedRegionOcr &&
            pageAnalysis.Regions.Any(
                x =>
                    x.Kind is
                        PageRegionKind.TextBubble or
                        PageRegionKind.TextFree))
        {
            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.RegionOcr,
                    "RT-DETR 텍스트 영역 1배·2배 OCR"));

            regionBatch =
                await ocr.RecognizeRegionsAsync(
                    sourcePath,
                    pageAnalysis.Regions,
                    progress,
                    token);

            regionLineDecisions =
                TextEvidenceFusionService.Validate(
                    pageAnalysis.Regions,
                    regionBatch.Observations,
                    observationBatch.Observations,
                    token);

            int acceptedRegionCount =
                regionLineDecisions.Count(
                    x => x.Accepted);

            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.OcrValidation,
                    $"독립 텍스트 영역 OCR · {regionBatch.Observations.Count}개 중 evidence 통과 {acceptedRegionCount}개"));
        }

        var containerBatch =
            await ocr.RecognizeContainersAsync(
                sourcePath,
                eligible,
                progress,
                token);

        token.ThrowIfCancellationRequested();

        var lineDecisions =
            ContainerOcrValidator.ValidateLines(
                eligible,
                containerBatch.Observations,
                token);

        var acceptedContainerIds =
            lineDecisions
                .Where(x => x.Accepted)
                .Select(x => x.ObservationId)
                .ToHashSet();

        var acceptedRegionIds =
            regionLineDecisions
                .Where(x => x.Accepted)
                .Select(x => x.ObservationId)
                .ToHashSet();

        var observations =
            observationBatch.Observations
                .Concat(regionBatch.Observations)
                .Concat(containerBatch.Observations)
                .ToList();

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.OcrValidation,
                $"컨테이너 관측 검증 · {containerBatch.Observations.Count}개 중 교차 검증 통과 {acceptedContainerIds.Count}개"));

        var lines =
            observationBatch.MergedLines
                .ToList();

        OcrEngine.MergeLines(
            lines,
            regionBatch.Observations
                .Where(
                    x =>
                        acceptedRegionIds.Contains(
                            x.ObservationId))
                .Select(
                    x => x.ToLine()));

        OcrEngine.MergeLines(
            lines,
            containerBatch.Observations
                .Where(
                    x =>
                        acceptedContainerIds.Contains(
                            x.ObservationId))
                .Select(
                    x => x.ToLine()));

        lines =
            lines
                .OrderBy(x => x.Y)
                .ThenBy(x => x.X)
                .ToList();

        if (lines.Count == 0)
        {
            return new OcrStageResult(
                pageCandidates,
                observations,
                lines,
                [],
                new OcrUnitBuildResult(
                    [],
                    0,
                    0,
                    0))
            {
                CandidateDecisions = candidateDecisions,
                ContainerAttempts = containerBatch.Attempts,
                ContainerLineDecisions = lineDecisions,
                RegionAttempts = regionBatch.Attempts,
                RegionLineDecisions = regionLineDecisions,
                PageAnalysis = pageAnalysis
            };
        }

        var preliminaryBlocks =
            blockGrouper.Group(
                lines);

        var unitBuild =
            await Task.Run(
                () =>
                    unitBuilder.Build(
                        lines,
                        pageCandidates,
                        candidateDecisions,
                        containerBatch.Observations,
                        lineDecisions,
                        token),
                token);

        return new OcrStageResult(
            pageCandidates,
            observations,
            lines,
            preliminaryBlocks,
            unitBuild)
        {
            CandidateDecisions = candidateDecisions,
            ContainerAttempts = containerBatch.Attempts,
            ContainerLineDecisions = lineDecisions,
            RegionAttempts = regionBatch.Attempts,
            RegionLineDecisions = regionLineDecisions,
            PageAnalysis = pageAnalysis
        };
    }

    public static string FormatPassSummary(
        IReadOnlyList<OcrObservation> observations)
    {
        var parts =
            observations
                .GroupBy(x => x.Pass)
                .OrderBy(x => x.Key)
                .Select(
                    x =>
                        $"{PassLabel(x.Key)} {x.Count()}")
                .ToList();

        return parts.Count == 0
            ? "관측 0"
            : string.Join(
                " / ",
                parts);
    }

    static string PassLabel(
        OcrPassKind pass)
        => pass switch
        {
            OcrPassKind.Global1x => "전체1x",
            OcrPassKind.Global2x => "전체2x",
            OcrPassKind.Focus3x => "focus3x",
            OcrPassKind.Region1x => "region1x",
            OcrPassKind.Region2x => "region2x",
            OcrPassKind.Container1x => "container1x",
            OcrPassKind.Container2x => "container2x",
            OcrPassKind.Container3x => "container3x",
            _ => pass.ToString()
        };
}
