using LocalMangaTranslator.Services;

namespace LocalMangaTranslator.Models;

public sealed record VisionTranslation(
    int Id,
    OcrLine Source,
    string CorrectedText,
    string Translation
);

public sealed class VisionTranslationDocument
{
    public string SourceFile { get; set; } = "";
    public string Model { get; set; } = "";
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    public List<VisionTranslation> Regions { get; set; } = [];
}
