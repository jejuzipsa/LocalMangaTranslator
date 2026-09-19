using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.Models;

/// <summary>
/// OCR is observation data, not final truth. Keeping the originating pass lets
/// later validation decide whether a line was seen once, repeatedly, globally,
/// or only inside a focused/container crop.
/// </summary>
public enum OcrPassKind
{
    Global1x,
    Global2x,
    Focus3x,

    // Reserved for the next container-first phase.
    Container1x,
    Container2x,
    Container3x
}

public sealed record OcrObservation(
    string ObservationId,
    OcrPassKind Pass,
    double X,
    double Y,
    double W,
    double H,
    string Text,
    float Confidence,
    string Language,
    double Scale,
    string SourceKey)
{
    public OcrLine ToLine()
        => new(
            X,
            Y,
            W,
            H,
            Text,
            Confidence,
            Language);
}

public sealed record OcrObservationBatch(
    IReadOnlyList<OcrObservation> Observations,
    IReadOnlyList<OcrLine> MergedLines);


public enum ContainerCandidateKind
{
    Speech,
    Caption,
    Unknown
}

/// <summary>
/// A page-wide geometric candidate. It is deliberately not a translation unit
/// and not erase permission. Later validation must attach text evidence before
/// a candidate can participate in translation or deletion.
/// </summary>
public sealed record ContainerCandidate(
    string CandidateId,
    ContainerCandidateKind Kind,
    Rect Bounds,
    string DetectorMode,
    bool Dark,
    double Score,
    double FillRatio,
    int BorderTouches,
    byte[] Mask,
    int MaskWidth,
    int MaskHeight);

public sealed record OcrUnitBuildResult(
    IReadOnlyList<OcrTextBlock> Units,
    int ContainerCount,
    int AssignedLineCount,
    int OrphanGroupCount);

/// <summary>
/// Explicit boundary between OCR acquisition and Vision review.
/// Page-wide container detection and container-specific OCR can be inserted
/// before this result is converted into translation units.
/// </summary>
public sealed record OcrStageResult(
    IReadOnlyList<ContainerCandidate> PageCandidates,
    IReadOnlyList<OcrObservation> Observations,
    IReadOnlyList<OcrLine> MergedLines,
    IReadOnlyList<OcrTextBlock> PreliminaryBlocks,
    OcrUnitBuildResult UnitBuild);

public enum PipelineStageKind
{
    OcrObservation,
    OcrUnitFormation,
    VisionReview,
    Translation,
    Render,
    Completed
}

public sealed record PipelineProgress(
    PipelineStageKind Stage,
    string Message);

public sealed record PagePipelineResult(
    bool HasText,
    OcrStageResult Ocr,
    IReadOnlyList<VisionTranslation> Reviewed,
    IReadOnlyList<VisionTranslation> Translated,
    string? TranslationJsonPath,
    string? OutputImagePath);
