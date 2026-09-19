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
        CancellationToken token = default)
    {
        var pageCandidates =
            await Task.Run(
                () => pageContainerDetector.Detect(
                    sourcePath,
                    token),
                token);

        var observationBatch =
            await ocr.RecognizeDetailedAsync(
                sourcePath,
                token);

        var lines =
            observationBatch.MergedLines
                .ToList();

        if (lines.Count == 0)
        {
            return new OcrStageResult(
                pageCandidates,
                observationBatch.Observations,
                lines,
                [],
                new OcrUnitBuildResult(
                    [],
                    0,
                    0,
                    0));
        }

        var preliminaryBlocks =
            blockGrouper.Group(lines);

        var unitBuild =
            await Task.Run(
                () => unitBuilder.Build(
                    sourcePath,
                    lines,
                    preliminaryBlocks,
                    token),
                token);

        return new OcrStageResult(
            pageCandidates,
            observationBatch.Observations,
            lines,
            preliminaryBlocks,
            unitBuild);
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
