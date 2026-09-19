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

    // Container passes retain the page candidate ID in SourceKey.
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

// OCR eligibility is geometry-only; it never grants erase permission.
public sealed record ContainerOcrDecision(string CandidateId, bool Eligible, string Reason);
public sealed record ContainerOcrAttempt(string CandidateId, OcrPassKind Pass, string Status, int Count, string? Error = null);
public sealed record ContainerOcrBatch(
    IReadOnlyList<OcrObservation> Observations,
    IReadOnlyList<ContainerOcrAttempt> Attempts);
public sealed record ContainerLineDecision(string ObservationId, string CandidateId, bool Accepted, string Reason, double MaskCoverage);


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
    OcrUnitBuildResult UnitBuild)
{
    public IReadOnlyList<ContainerOcrDecision> CandidateDecisions { get; init; } = [];
    public IReadOnlyList<ContainerOcrAttempt> ContainerAttempts { get; init; } = [];
    public IReadOnlyList<ContainerLineDecision> ContainerLineDecisions { get; init; } = [];
}

public enum PipelineStageKind
{
    ContainerDetection,
    ContainerValidation,
    ContainerOcr,
    OcrValidation,
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
