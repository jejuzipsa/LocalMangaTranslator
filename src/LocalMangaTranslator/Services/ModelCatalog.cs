using System.Text.Json;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class ModelCatalog
{
    readonly string root;

    public ModelCatalog(string root) => this.root = root;

    public IReadOnlyList<ModelProfile> Load()
    {
        var result = new List<ModelProfile>();
        if (!Directory.Exists(root))
            return result;

        foreach (var manifest in Directory.EnumerateFiles(root, "model.json", SearchOption.AllDirectories))
        {
            try
            {
                var profile = JsonSerializer.Deserialize<ModelProfile>(
                    File.ReadAllText(manifest),
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (profile is not null && profile.Enabled && !string.IsNullOrWhiteSpace(profile.Id))
                    result.Add(profile);
            }
            catch
            {
                // 잘못된 manifest 하나 때문에 프로그램 전체가 멈추지 않게 한다.
            }
        }

        return result.OrderBy(x => x.Name, StringComparer.CurrentCultureIgnoreCase).ToList();
    }

    public IReadOnlyList<ModelProfile> LoadForTask(string task)
        => Load()
            .Where(x => x.SupportsTask(task))
            .OrderBy(x => x.Name, StringComparer.CurrentCultureIgnoreCase)
            .ToList();
}
