using LocalMangaTranslator.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Apache-2.0 RT-DETR-v2 detector published by ogkalu.
/// The model is trained specifically for comic bubble/text detection and is
/// intentionally independent from OCR recognition.
/// </summary>
public sealed class RtdetrPageRegionAnalyzer : IDisposable
{
    const int InputSize = 640;
    const float DefaultThreshold = 0.30f;

    readonly object sync = new();
    InferenceSession? session;

    public bool IsReady =>
        ExternalModelManager.IsRtdetrReady();

    public IReadOnlyList<PageRegion> Analyze(
        string sourcePath,
        CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested();

        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "RT-DETR 분석을 위해 원본 이미지를 열 수 없습니다.");

        var runtime =
            GetSession();

        using var resized =
            new Mat();

        Cv2.Resize(
            source,
            resized,
            new Size(
                InputSize,
                InputSize),
            0,
            0,
            InterpolationFlags.Linear);

        var imageTensor =
            new DenseTensor<float>(
                new[]
                {
                    1,
                    3,
                    InputSize,
                    InputSize
                });

        for (int y = 0; y < InputSize; y++)
        {
            token.ThrowIfCancellationRequested();

            for (int x = 0; x < InputSize; x++)
            {
                var bgr =
                    resized.At<Vec3b>(
                        y,
                        x);

                imageTensor[0, 0, y, x] =
                    bgr.Item2 / 255f;

                imageTensor[0, 1, y, x] =
                    bgr.Item1 / 255f;

                imageTensor[0, 2, y, x] =
                    bgr.Item0 / 255f;
            }
        }

        // This export follows the Comic Translate ONNX wrapper:
        // [width, height] is supplied to orig_target_sizes and the model returns
        // boxes already projected to source-image coordinates.
        var sizeTensor =
            new DenseTensor<long>(
                new[]
                {
                    1,
                    2
                });

        sizeTensor[0, 0] =
            source.Cols;

        sizeTensor[0, 1] =
            source.Rows;

        var inputs =
            new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(
                    "images",
                    imageTensor),

                NamedOnnxValue.CreateFromTensor(
                    "orig_target_sizes",
                    sizeTensor)
            };

        using var results =
            runtime.Run(inputs);

        var labelsResult =
            FindOutput(
                results,
                "labels",
                0);

        var boxesResult =
            FindOutput(
                results,
                "boxes",
                1);

        var scoresResult =
            FindOutput(
                results,
                "scores",
                2);

        long[] labels =
            labelsResult.AsTensor<long>()
                .ToArray();

        float[] boxes =
            boxesResult.AsTensor<float>()
                .ToArray();

        float[] scores =
            scoresResult.AsTensor<float>()
                .ToArray();

        int count =
            Math.Min(
                labels.Length,
                Math.Min(
                    scores.Length,
                    boxes.Length / 4));

        var raw =
            new List<PageRegion>();

        for (int i = 0; i < count; i++)
        {
            token.ThrowIfCancellationRequested();

            float score =
                scores[i];

            if (!float.IsFinite(score) ||
                score < DefaultThreshold)
            {
                continue;
            }

            var kind =
                labels[i] switch
                {
                    0 => PageRegionKind.Bubble,
                    1 => PageRegionKind.TextBubble,
                    2 => PageRegionKind.TextFree,
                    _ => PageRegionKind.Unknown
                };

            if (kind == PageRegionKind.Unknown)
                continue;

            int offset =
                i * 4;

            int left =
                (int)Math.Floor(
                    Math.Min(
                        boxes[offset],
                        boxes[offset + 2]));

            int top =
                (int)Math.Floor(
                    Math.Min(
                        boxes[offset + 1],
                        boxes[offset + 3]));

            int right =
                (int)Math.Ceiling(
                    Math.Max(
                        boxes[offset],
                        boxes[offset + 2]));

            int bottom =
                (int)Math.Ceiling(
                    Math.Max(
                        boxes[offset + 1],
                        boxes[offset + 3]));

            var bounds =
                ClampRect(
                    left,
                    top,
                    right - left,
                    bottom - top,
                    source.Cols,
                    source.Rows);

            if (bounds.Width < 4 ||
                bounds.Height < 4)
            {
                continue;
            }

            raw.Add(
                new PageRegion(
                    "",
                    kind,
                    bounds,
                    score,
                    "ogkalu-rtdetr-v2"));
        }

        var deduplicated =
            Deduplicate(
                raw);

        return deduplicated
            .Select(
                (x, index) =>
                    x with
                    {
                        RegionId =
                            $"RG{index + 1:000}"
                    })
            .ToList();
    }

    InferenceSession GetSession()
    {
        lock (sync)
        {
            if (session is not null)
                return session;

            if (!IsReady)
            {
                throw new InvalidOperationException(
                    "RT-DETR 페이지 분석 모델이 설치되어 있지 않습니다.");
            }

            var options =
                new SessionOptions
                {
                    GraphOptimizationLevel =
                        GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode =
                        ExecutionMode.ORT_SEQUENTIAL,
                    IntraOpNumThreads =
                        Math.Clamp(
                            Environment.ProcessorCount / 2,
                            2,
                            6),
                    InterOpNumThreads = 1
                };

            session =
                new InferenceSession(
                    ExternalModelManager.RtdetrModelPath,
                    options);

            ValidateSignature(
                session);

            return session;
        }
    }

    static DisposableNamedOnnxValue FindOutput(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string name,
        int fallbackIndex)
    {
        var named =
            results.FirstOrDefault(
                x => string.Equals(
                    x.Name,
                    name,
                    StringComparison.OrdinalIgnoreCase));

        if (named is not null)
            return named;

        return results.ElementAt(
            fallbackIndex);
    }

    static void ValidateSignature(
        InferenceSession runtime)
    {
        if (!runtime.InputMetadata.ContainsKey("images") ||
            !runtime.InputMetadata.ContainsKey("orig_target_sizes"))
        {
            throw new InvalidOperationException(
                "RT-DETR ONNX 입력 시그니처가 예상과 다릅니다.");
        }
    }

    static List<PageRegion> Deduplicate(
        IReadOnlyList<PageRegion> input)
    {
        var kept =
            new List<PageRegion>();

        foreach (var candidate in
                 input
                     .OrderByDescending(x => x.Score)
                     .ThenByDescending(
                         x =>
                             x.Bounds.Width *
                             (long)x.Bounds.Height))
        {
            bool duplicate =
                kept.Any(
                    x =>
                        x.Kind == candidate.Kind &&
                        IoU(
                            x.Bounds,
                            candidate.Bounds) >=
                        0.78);

            if (!duplicate)
                kept.Add(candidate);
        }

        return kept
            .OrderBy(x => x.Bounds.Y)
            .ThenBy(x => x.Bounds.X)
            .ToList();
    }

    static double IoU(
        Rect a,
        Rect b)
    {
        int left =
            Math.Max(
                a.Left,
                b.Left);

        int top =
            Math.Max(
                a.Top,
                b.Top);

        int right =
            Math.Min(
                a.Right,
                b.Right);

        int bottom =
            Math.Min(
                a.Bottom,
                b.Bottom);

        double intersection =
            Math.Max(
                0,
                right - left) *
            (double)Math.Max(
                0,
                bottom - top);

        if (intersection <= 0)
            return 0;

        double areaA =
            Math.Max(
                1,
                a.Width * (double)a.Height);

        double areaB =
            Math.Max(
                1,
                b.Width * (double)b.Height);

        return intersection /
               Math.Max(
                   1,
                   areaA +
                   areaB -
                   intersection);
    }

    static Rect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left =
            Math.Clamp(
                x,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                y,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                x + width,
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                y + height,
                0,
                imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(
                0,
                right - left),
            Math.Max(
                0,
                bottom - top));
    }

    public void Dispose()
    {
        lock (sync)
        {
            session?.Dispose();
            session = null;
        }
    }
}
