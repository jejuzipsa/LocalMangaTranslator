namespace LocalMangaTranslator.Models;

public sealed class ModelProfile
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Type { get; set; } = "vision_llm";
    public string Provider { get; set; } = "ollama";
    public string ModelTag { get; set; } = "";
    public string ApiBase { get; set; } = "http://127.0.0.1:11434";
    public bool SupportsImage { get; set; } = true;
    public bool SupportsText { get; set; } = true;
    public bool Enabled { get; set; } = true;

    public override string ToString() => Name;
}
