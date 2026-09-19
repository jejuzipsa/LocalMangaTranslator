using System.Net.Http;
using System.Security.Cryptography;

namespace LocalMangaTranslator.Services;

public static class ExternalModelManager
{
    const string RtdetrFileName = "detector-v4-s_int8.onnx";
    const string RtdetrUrl =
        "https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/" +
        RtdetrFileName + "?download=true";

    // Known upstream revisions of the Apache-2.0 INT8 detector.
    static readonly HashSet<string> KnownRtdetrSha256 =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "fb9d4f56c68c7345113b81909139171f1e2fbec2aa7f837768a897fb583a3a4f",
            "5fe9e4f576e49d4e7e8b0e029d6d3cdc252abd4694113e1cae120e62c931ea79"
        };

    static readonly HttpClient Http = new()
    {
        Timeout = TimeSpan.FromMinutes(30)
    };

    const string BaberuBaseUrl =
        "https://huggingface.co/genshiai-daichi/baberu-ocr/resolve/main/";

    static readonly (string RelativePath, long MinimumBytes)[] BaberuFiles =
    [
        ("onnx/vision_int4.onnx", 45L * 1024 * 1024),
        ("onnx/decoder_prefill_int8.onnx", 30L * 1024 * 1024),
        ("onnx/decoder_step_int8.onnx", 30L * 1024 * 1024),
        ("tokenizer/vocab.json", 100L * 1024)
    ];

    public static string RtdetrModelPath
    {
        get
        {
            string? overridePath =
                Environment.GetEnvironmentVariable(
                    "LMT_RTDETR_MODEL_PATH");

            if (!string.IsNullOrWhiteSpace(overridePath))
                return overridePath;

            return Path.Combine(
                AppContext.BaseDirectory,
                "models",
                "layout",
                "comic-text-bubble",
                RtdetrFileName);
        }
    }

    public static bool IsRtdetrReady()
        => File.Exists(RtdetrModelPath) &&
           new FileInfo(RtdetrModelPath).Length > 8 * 1024 * 1024;

    public static async Task<string> EnsureRtdetrAsync(
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (IsRtdetrReady())
        {
            progress?.Report("RT-DETR 모델 준비됨");
            return RtdetrModelPath;
        }

        Directory.CreateDirectory(
            Path.GetDirectoryName(RtdetrModelPath)!);

        string temporary =
            RtdetrModelPath + ".download";

        try
        {
            progress?.Report("RT-DETR 만화 페이지 분석 모델 다운로드 시작");

            using var response =
                await Http.GetAsync(
                    RtdetrUrl,
                    HttpCompletionOption.ResponseHeadersRead,
                    token);

            response.EnsureSuccessStatusCode();

            long? total =
                response.Content.Headers.ContentLength;

            long received = 0;
            int lastReportedPercent = -1;

            await using (var input =
                         await response.Content.ReadAsStreamAsync(token))
            {
                await using (var output =
                             new FileStream(
                                 temporary,
                                 FileMode.Create,
                                 FileAccess.Write,
                                 FileShare.None,
                                 1024 * 1024,
                                 useAsync: true))
                {
                    var buffer =
                        new byte[1024 * 1024];

                    while (true)
                    {
                        int read =
                            await input.ReadAsync(
                                buffer.AsMemory(),
                                token);

                        if (read <= 0)
                            break;

                        await output.WriteAsync(
                            buffer.AsMemory(0, read),
                            token);

                        received += read;

                        if (total is > 0)
                        {
                            int percent =
                                (int)Math.Clamp(
                                    Math.Floor(
                                        received * 100.0 /
                                        total.Value),
                                    0,
                                    100);

                            if (percent != lastReportedPercent)
                            {
                                lastReportedPercent = percent;
                                progress?.Report(
                                    $"RT-DETR 다운로드 {percent}%");
                            }
                        }
                    }

                    await output.FlushAsync(token);
                }
            }

            // The download stream must be fully disposed before the temporary
            // file is reopened for hashing or moved into its final location.
            if (received < 8 * 1024 * 1024)
                throw new InvalidOperationException(
                    "RT-DETR 모델 파일이 예상보다 작습니다.");

            string hash =
                await ComputeSha256Async(
                    temporary,
                    token);

            if (!KnownRtdetrSha256.Contains(hash))
            {
                throw new InvalidOperationException(
                    "RT-DETR 모델 체크섬이 알려진 공식 파일과 다릅니다. " +
                    $"SHA256={hash}");
            }

            File.Move(
                temporary,
                RtdetrModelPath,
                overwrite: true);

            progress?.Report(
                $"RT-DETR 모델 설치 완료 · SHA256 {hash[..12]}…");

            return RtdetrModelPath;
        }
        finally
        {
            try
            {
                if (File.Exists(temporary))
                    File.Delete(temporary);
            }
            catch
            {
            }
        }
    }

    public static string BaberuModelRoot =>
        Path.Combine(
            AppContext.BaseDirectory,
            "models",
            "ocr",
            "baberu");

    public static string BaberuVisionPath =>
        Path.Combine(
            BaberuModelRoot,
            "onnx",
            "vision_int4.onnx");

    public static string BaberuPrefillPath =>
        Path.Combine(
            BaberuModelRoot,
            "onnx",
            "decoder_prefill_int8.onnx");

    public static string BaberuStepPath =>
        Path.Combine(
            BaberuModelRoot,
            "onnx",
            "decoder_step_int8.onnx");

    public static string BaberuVocabPath =>
        Path.Combine(
            BaberuModelRoot,
            "tokenizer",
            "vocab.json");

    public static bool IsBaberuReady()
        => BaberuFiles.All(file =>
        {
            string path =
                Path.Combine(
                    BaberuModelRoot,
                    file.RelativePath.Replace('/', Path.DirectorySeparatorChar));

            return File.Exists(path) &&
                   new FileInfo(path).Length >=
                   file.MinimumBytes;
        });

    public static async Task<string> EnsureBaberuAsync(
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (IsBaberuReady())
        {
            progress?.Report("Baberu OCR 모델 준비됨");
            return BaberuModelRoot;
        }

        for (int i = 0; i < BaberuFiles.Length; i++)
        {
            var file =
                BaberuFiles[i];

            string target =
                Path.Combine(
                    BaberuModelRoot,
                    file.RelativePath.Replace('/', Path.DirectorySeparatorChar));

            if (File.Exists(target) &&
                new FileInfo(target).Length >= file.MinimumBytes)
            {
                continue;
            }

            Directory.CreateDirectory(
                Path.GetDirectoryName(target)!);

            string label =
                Path.GetFileName(target);

            progress?.Report(
                $"Baberu OCR {i + 1}/{BaberuFiles.Length} · {label}");

            await DownloadFileAsync(
                BaberuBaseUrl +
                file.RelativePath +
                "?download=true",
                target,
                file.MinimumBytes,
                "Baberu OCR",
                progress,
                token);
        }

        if (!IsBaberuReady())
        {
            throw new InvalidOperationException(
                "Baberu OCR 모델 설치가 완료되지 않았습니다.");
        }

        progress?.Report("Baberu OCR 모델 설치 완료");
        return BaberuModelRoot;
    }

    static async Task DownloadFileAsync(
        string url,
        string target,
        long minimumBytes,
        string label,
        IProgress<string>? progress,
        CancellationToken token)
    {
        string temporary =
            target + ".download";

        try
        {
            using var response =
                await Http.GetAsync(
                    url,
                    HttpCompletionOption.ResponseHeadersRead,
                    token);

            response.EnsureSuccessStatusCode();

            long? total =
                response.Content.Headers.ContentLength;

            long received = 0;
            int lastReportedPercent = -1;

            await using (var input =
                         await response.Content.ReadAsStreamAsync(token))
            {
                await using var output =
                    new FileStream(
                        temporary,
                        FileMode.Create,
                        FileAccess.Write,
                        FileShare.None,
                        1024 * 1024,
                        useAsync: true);

                var buffer =
                    new byte[1024 * 1024];

                while (true)
                {
                    int read =
                        await input.ReadAsync(
                            buffer.AsMemory(),
                            token);

                    if (read <= 0)
                        break;

                    await output.WriteAsync(
                        buffer.AsMemory(0, read),
                        token);

                    received += read;

                    if (total is > 0)
                    {
                        int percent =
                            (int)Math.Clamp(
                                Math.Floor(
                                    received * 100.0 /
                                    total.Value),
                                0,
                                100);

                        if (percent != lastReportedPercent)
                        {
                            lastReportedPercent = percent;
                            progress?.Report(
                                $"{label} 다운로드 {percent}% · {Path.GetFileName(target)}");
                        }
                    }
                }

                await output.FlushAsync(token);
            }

            if (received < minimumBytes)
            {
                throw new InvalidOperationException(
                    $"{label} 파일이 예상보다 작습니다: {Path.GetFileName(target)}");
            }

            string hash =
                await ComputeSha256Async(
                    temporary,
                    token);

            File.Move(
                temporary,
                target,
                overwrite: true);

            progress?.Report(
                $"{label} 준비 · {Path.GetFileName(target)} · SHA256 {hash[..12]}…");
        }
        finally
        {
            try
            {
                if (File.Exists(temporary))
                    File.Delete(temporary);
            }
            catch
            {
            }
        }
    }

    static async Task<string> ComputeSha256Async(
        string path,
        CancellationToken token)
    {
        await using var stream =
            new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                1024 * 1024,
                useAsync: true);

        using var sha =
            SHA256.Create();

        byte[] hash =
            await sha.ComputeHashAsync(
                stream,
                token);

        return Convert.ToHexString(hash)
            .ToLowerInvariant();
    }
}
