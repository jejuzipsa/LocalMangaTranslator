using System.Text;
using System.Text.Json;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class VisionTranslationService
{
    readonly HttpClient http = new()
    {
        Timeout = TimeSpan.FromMinutes(10)
    };

    public async Task<List<VisionTranslation>> ReviewAndTranslateAsync(
        string imagePath,
        IReadOnlyList<OcrLine> ocrLines,
        ModelProfile model,
        CancellationToken token = default)
    {
        if (ocrLines.Count == 0) return [];

        var ocrPayload = ocrLines.Select((x, id) => new
        {
            id,
            x = Math.Round(x.X, 1),
            y = Math.Round(x.Y, 1),
            width = Math.Round(x.W, 1),
            height = Math.Round(x.H, 1),
            ocr = x.Text,
            confidence = Math.Round(x.Confidence, 3),
            language = x.Language
        }).ToArray();

        var prompt = BuildPrompt(ocrPayload);
        var imageBase64 = Convert.ToBase64String(await File.ReadAllBytesAsync(imagePath, token));

        var request = new
        {
            model = model.ModelTag,
            stream = false,
            format = "json",
            options = new { temperature = 0.1 },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = prompt,
                    images = new[] { imageBase64 }
                }
            }
        };

        using var message = new HttpRequestMessage(
            HttpMethod.Post,
            $"{model.ApiBase.TrimEnd('/')}/api/chat")
        {
            Content = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json")
        };

        using var response = await http.SendAsync(message, token);
        var responseText = await response.Content.ReadAsStringAsync(token);

        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException(
                $"Ollama 응답 오류 {(int)response.StatusCode}: {Compact(responseText)}");

        using var outer = JsonDocument.Parse(responseText);
        if (!outer.RootElement.TryGetProperty("message", out var messageEl) ||
            !messageEl.TryGetProperty("content", out var contentEl))
            throw new InvalidOperationException("Ollama 응답에 message.content가 없습니다.");

        var content = contentEl.GetString();
        if (string.IsNullOrWhiteSpace(content))
            throw new InvalidOperationException("Vision LLM 응답이 비어 있습니다.");

        var parsed = ParseResult(content, ocrLines);
        if (parsed.Count == 0)
            throw new InvalidOperationException($"Vision LLM JSON을 해석하지 못했습니다: {Compact(content)}");

        return parsed;
    }

    static string BuildPrompt(object ocrPayload)
    {
        var json = JsonSerializer.Serialize(ocrPayload);

        return $"""
너는 일본어/영어 만화 이미지 OCR 검수 및 한국어 번역기다.
첨부된 원본 이미지와 아래 OCR 결과를 함께 확인하라.

목표:
1. OCR 텍스트가 이미지와 다르면 corrected에 정확한 원문을 적는다.
2. OCR이 맞으면 corrected에 동일한 원문을 적는다.
3. translation에는 자연스러운 한국어 번역만 적는다.
4. 말투, 호칭, 감정, 문장부호를 원문 맥락에 맞게 유지한다.
5. 효과음/의성어가 OCR에 포함되어 있으면 의미를 살려 번역한다.
6. 항목을 추가/삭제/병합하지 말고 반드시 입력 id를 그대로 유지한다.
7. 설명이나 마크다운 없이 JSON만 출력한다.

출력 형식:
{{
  "regions": [
    {{ "id": 0, "corrected": "원문", "translation": "한국어" }}
  ]
}}

OCR:
{json}
""";
    }

    static List<VisionTranslation> ParseResult(string content, IReadOnlyList<OcrLine> lines)
    {
        try
        {
            var normalized = content.Trim();
            var fenceToken = new string((char)96, 3);
            if (normalized.StartsWith(fenceToken))
            {
                var firstNewLine = normalized.IndexOf('\n');
                if (firstNewLine >= 0) normalized = normalized[(firstNewLine + 1)..];
                var fence = normalized.LastIndexOf(fenceToken, StringComparison.Ordinal);
                if (fence >= 0) normalized = normalized[..fence].Trim();
            }

            using var doc = JsonDocument.Parse(normalized);
            var root = doc.RootElement;
            if (!root.TryGetProperty("regions", out var regions) ||
                regions.ValueKind != JsonValueKind.Array)
                return [];

            var result = new List<VisionTranslation>();
            var used = new HashSet<int>();

            foreach (var region in regions.EnumerateArray())
            {
                if (!region.TryGetProperty("id", out var idEl) ||
                    !idEl.TryGetInt32(out var id))
                    continue;

                if (id < 0 || id >= lines.Count || !used.Add(id))
                    continue;

                var corrected = region.TryGetProperty("corrected", out var correctedEl)
                    ? correctedEl.GetString()?.Trim()
                    : null;

                var translation = region.TryGetProperty("translation", out var translationEl)
                    ? translationEl.GetString()?.Trim()
                    : null;

                result.Add(new VisionTranslation(
                    id,
                    lines[id],
                    string.IsNullOrWhiteSpace(corrected) ? lines[id].Text : corrected,
                    string.IsNullOrWhiteSpace(translation)
                        ? corrected ?? lines[id].Text
                        : translation));
            }

            return result.OrderBy(x => x.Id).ToList();
        }
        catch
        {
            return [];
        }
    }

    static string Compact(string text)
    {
        var oneLine = text.Replace('\r', ' ').Replace('\n', ' ').Trim();
        return oneLine.Length <= 500 ? oneLine : oneLine[..500] + "...";
    }
}
