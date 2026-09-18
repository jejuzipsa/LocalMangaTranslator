using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.Models;

public sealed record PlannedOcrLine(
    string LineId,
    OcrLine Source);

public sealed record TextLayoutPlan(
    bool Fits,
    string Reason,
    string Text,
    double BoxX,
    double BoxY,
    double BoxWidth,
    double BoxHeight,
    double FontSize,
    double LineHeightFactor,
    string Alignment,
    double OriginX,
    double OriginY,
    double GlyphX,
    double GlyphY,
    double GlyphWidth,
    double GlyphHeight,
    double OutlineWidth);

public sealed record ErasePlan(
    string UnitId,
    string ContainerId,
    Rect AllowedBounds,
    byte[] AllowedMask,
    int MaskWidth,
    int MaskHeight,
    IReadOnlyList<string> LineIds,
    IReadOnlyList<string> TemporaryDetachedLineIds);

public sealed record RenderUnitPlan(
    string UnitId,
    string ContainerId,
    VisionTranslation Region,
    BalloonLayout Container,
    IReadOnlyList<PlannedOcrLine> Lines,
    TextLayoutPlan Layout,
    bool Approved,
    string Reason);

public sealed class RenderPlanDocument
{
    public string SourceFile { get; set; } = "";
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    public List<RenderPlanDiagnostic> Units { get; set; } = [];
}

public sealed class RenderPlanDiagnostic
{
    public string UnitId { get; set; } = "";
    public string ContainerId { get; set; } = "";
    public int RegionId { get; set; }
    public List<string> LineIds { get; set; } = [];
    public string Type { get; set; } = "";
    public bool ContainerDetected { get; set; }
    public string ContainerMode { get; set; } = "";
    public bool Approved { get; set; }
    public string Reason { get; set; } = "";
    public bool LayoutFits { get; set; }
    public string LayoutReason { get; set; } = "";
    public double FontSize { get; set; }
    public double BoxX { get; set; }
    public double BoxY { get; set; }
    public double BoxWidth { get; set; }
    public double BoxHeight { get; set; }
    public double GlyphX { get; set; }
    public double GlyphY { get; set; }
    public double GlyphWidth { get; set; }
    public double GlyphHeight { get; set; }
}
