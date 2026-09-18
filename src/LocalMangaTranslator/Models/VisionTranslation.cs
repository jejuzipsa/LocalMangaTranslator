namespace LocalMangaTranslator.Models;

public sealed record VisionTranslation(
    int Id,
    OcrTextBlock Source,
    string CorrectedText,
    string Translation,
    string Type,
    bool Render
);

public sealed class VisionTranslationDocument
{
    public string SourceFile { get; set; } = "";
    public string Model { get; set; } = "";
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    public List<VisionTranslation> Regions { get; set; } = [];
}
