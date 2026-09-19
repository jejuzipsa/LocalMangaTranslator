using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Owns the OCR-side stages of the translation pipeline.
/// MainWindow should not need to know how many OCR passes, grouping passes,
/// or ownership passes are required to produce Vision-ready units.
/// </summary>
public sealed class OcrPipelineService
{
    readonly OcrEngine ocr;
    readonly PageContainerCandidateDetector pageContainerDetector = new();
    readonly OcrBlockGrouper blockGrouper = new();
    readonly OcrContainerUnitBuilder unitBuilder = new();

    public OcrPipelineService(
        OcrEngine ocr)
    {
        this.ocr = ocr;
    }

    public async Task<OcrStageResult> AnalyzeAsync(
        string sourcePath,
        CancellationToken token = default,
        IProgress<PipelineProgress>? progress = null)
    {
        progress?.Report(new(PipelineStageKind.ContainerDetection, "페이지 전체 컨테이너 후보 검출"));
        var pageCandidates =
            await Task.Run(
                () => pageContainerDetector.Detect(
                    sourcePath,
                    token),
                token);

        var candidateDecisions = pageCandidates.Select(ContainerOcrValidator.ValidateCandidate).ToList();
        var eligibleIds = candidateDecisions.Where(x => x.Eligible).Select(x => x.CandidateId).ToHashSet();
        var eligible = pageCandidates.Where(x => eligibleIds.Contains(x.CandidateId)).ToList();
        progress?.Report(new(PipelineStageKind.ContainerValidation,
            $"컨테이너 후보 검증 · {pageCandidates.Count}개 중 OCR 대상 {eligible.Count}개"));
        progress?.Report(new(PipelineStageKind.OcrObservation, "전체 1배·2배 / 집중 3배 OCR"));
        var observationBatch =
            await ocr.RecognizeDetailedAsync(
                sourcePath,
                token);

        var containerBatch = await ocr.RecognizeContainersAsync(sourcePath, eligible, progress, token);
        token.ThrowIfCancellationRequested();
        var lineDecisions = ContainerOcrValidator.ValidateLines(eligible, containerBatch.Observations, token);
        var acceptedIds = lineDecisions.Where(x => x.Accepted).Select(x => x.ObservationId).ToHashSet();
        var observations = observationBatch.Observations.Concat(containerBatch.Observations).ToList();
        progress?.Report(new(PipelineStageKind.OcrValidation,
            $"컨테이너 관측 검증 · {containerBatch.Observations.Count}개 중 교차 검증 통과 {acceptedIds.Count}개"));
        var lines =
            observationBatch.MergedLines
                .ToList();

        OcrEngine.MergeLines(lines, containerBatch.Observations
            .Where(x => acceptedIds.Contains(x.ObservationId)).Select(x => x.ToLine()));
        lines = lines.OrderBy(x => x.Y).ThenBy(x => x.X).ToList();

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
                ContainerLineDecisions = lineDecisions
            };
        }

        var preliminaryBlocks =
            blockGrouper.Group(lines);

        var unitBuild =
            await Task.Run(
                () => unitBuilder.Build(
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
            ContainerLineDecisions = lineDecisions
        };
    }

    public static string FormatPassSummary(
        IReadOnlyList<OcrObservation> observations)
    {
        var parts = observations
            .GroupBy(x => x.Pass)
            .OrderBy(x => x.Key)
            .Select(x =>
                $"{PassLabel(x.Key)} {x.Count()}")
            .ToList();

        return parts.Count == 0
            ? "관측 0"
            : string.Join(" / ", parts);
    }

    static string PassLabel(
        OcrPassKind pass)
        => pass switch
        {
            OcrPassKind.Global1x => "전체1x",
            OcrPassKind.Global2x => "전체2x",
            OcrPassKind.Focus3x => "focus3x",
            OcrPassKind.Container1x => "container1x",
            OcrPassKind.Container2x => "container2x",
            OcrPassKind.Container3x => "container3x",
            _ => pass.ToString()
        };
}

