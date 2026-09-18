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
);
