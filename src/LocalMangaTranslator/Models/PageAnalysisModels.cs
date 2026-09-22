using OpenCvSharp;

namespace LocalMangaTranslator.Models;

public enum RegionAnalysisMode
{
    Legacy,
    HybridRtdetr
}

public enum PipelineDecisionTrack
{
    Classic,
    LayaExperimental
}

public enum LayaDecisionMode
{
    Off,
    Shadow,
    Advisory,
    Active
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
    bool EnableFinalAudit = true,
    PipelineDecisionTrack DecisionTrack = PipelineDecisionTrack.LayaExperimental,
    LayaDecisionMode LayaMode = LayaDecisionMode.Advisory);

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
