using OpenCvSharp;

namespace LocalMangaTranslator.Models;

public enum RegionAnalysisMode
{
    Legacy,
    HybridRtdetr
}

public enum PageRegionKind
{
    Bubble,
    TextBubble,
    TextFree,
    Panel,
    Caption,
    Unknown
}

public sealed record PipelineOptions(
    RegionAnalysisMode RegionAnalysis = RegionAnalysisMode.HybridRtdetr,
    bool EnableTargetedRegionOcr = false,
    bool EnableBaberuOcr = true,
    bool EnableFinalAudit = true);

public sealed record PageRegion(
    string RegionId,
    PageRegionKind Kind,
    Rect Bounds,
    float Score,
    string Source);

public sealed record PageAnalysisResult(
    IReadOnlyList<PageRegion> Regions,
    IReadOnlyList<ContainerCandidate> ContainerCandidates,
    string Mode,
    bool ExternalDetectorUsed,
    string? ExternalDetectorStatus = null);
