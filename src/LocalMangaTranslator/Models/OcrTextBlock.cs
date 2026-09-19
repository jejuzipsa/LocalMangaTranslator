using System.Text.Json.Serialization;
using LocalMangaTranslator.Services;

namespace LocalMangaTranslator.Models;

public sealed record OcrTextBlock(
    int Id,
    double X,
    double Y,
    double W,
    double H,
    string Text,
    int OriginalRegionCount,
    string Language,
    IReadOnlyList<OcrLine> Lines
)
{
    public string? SecondaryOcrText { get; init; }
    public string? SecondaryOcrSource { get; init; }
    public string? SecondaryOcrAgreement { get; init; }
    public string? RegionId { get; init; }

    // Runtime-only structural evidence. The mask can be large, so keep it out
    // of page translation JSON while preserving it through Vision/Render.
    [JsonIgnore]
    public ContainerCandidate? RegionContainer { get; init; }

    // RT-DETR's paired TextBubble region. This is runtime-only structural
    // evidence used to rescue a line/container association and to constrain
    // erase permission more tightly than a whole balloon rectangle.
    [JsonIgnore]
    public PageRegion? RegionTextRegion { get; init; }
}