using System.Net.Http;
using System.Text;
using System.Text.Json;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class OllamaClient
{
    readonly HttpClient http = new()
    {
        Timeout = TimeSpan.FromMinutes(15)
    };

    public async Task<bool> IsServerReadyAsync(ModelProfile model, CancellationToken token = default)
    {
        try
        {
            using var response = await http.GetAsync($"{model.ApiBase.TrimEnd('/')}/api/tags", token);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<bool> IsModelInstalledAsync(ModelProfile model, CancellationToken token = default)
    {
        try
        {
            using var response = await http.GetAsync($"{model.ApiBase.TrimEnd('/')}/api/tags", token);
            if (!response.IsSuccessStatusCode) return false;

            using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(token));
            if (!doc.RootElement.TryGetProperty("models", out var models)) return false;

            foreach (var item in models.EnumerateArray())
            {
                var name = item.TryGetProperty("name", out var nameEl) ? nameEl.GetString() : null;
                var tag = item.TryGetProperty("model", out var modelEl) ? modelEl.GetString() : null;
                if (string.Equals(name, model.ModelTag, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(tag, model.ModelTag, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    public async Task PullModelAsync(
        ModelProfile model,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        var body = JsonSerializer.Serialize(new { model = model.ModelTag, stream = true });
        using var request = new HttpRequestMessage(
            HttpMethod.Post,
            $"{model.ApiBase.TrimEnd('/')}/api/pull")
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        };

        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, token);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(token);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            token.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync(token);
            if (string.IsNullOrWhiteSpace(line)) continue;

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;

            if (root.TryGetProperty("error", out var error))
                throw new InvalidOperationException(error.GetString() ?? "Ollama 모델 다운로드 오류");

            var status = root.TryGetProperty("status", out var statusEl)
                ? statusEl.GetString() ?? "처리 중"
                : "처리 중";

            long total = root.TryGetProperty("total", out var totalEl) && totalEl.TryGetInt64(out var t) ? t : 0;
            long completed = root.TryGetProperty("completed", out var doneEl) && doneEl.TryGetInt64(out var d) ? d : 0;

            if (total > 0 && completed >= 0)
            {
                var percent = Math.Clamp(completed * 100.0 / total, 0, 100);
                progress?.Report($"{status} · {FormatBytes(completed)} / {FormatBytes(total)} ({percent:0.0}%)");
            }
            else
            {
                progress?.Report(status);
            }
        }
    }

    static string FormatBytes(long bytes)
    {
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        return $"{value:0.##} {units[unit]}";
    }
}
