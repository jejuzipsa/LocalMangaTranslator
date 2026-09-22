using LocalMangaTranslator.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public sealed record RtdetrShadowRetryWindow(
    string PassId,
    Rect CropBounds,
    double ContextMultiplier,
    double InputPixelsPerSourcePixel);

public sealed record RtdetrShadowRetryPass(
    RtdetrShadowRetryWindow Window,
    IReadOnlyList<PageRegion> Regions);

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

        return AnalyzeMat(
            source,
            new Point(0, 0),
            "ogkalu-rtdetr-v2",
            "RG",
            token);
    }

    public IReadOnlyList<RtdetrShadowRetryPass> AnalyzeShadowRetry(
        string sourcePath,
        OcrTextBlock block,
        CancellationToken token = default)
    {
        token.ThrowIfCancellationRequested();

        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "RT-DETR Shadow retry를 위해 원본 이미지를 열 수 없습니다.");

        var windows =
            BuildShadowRetryWindows(
                source.Cols,
                source.Rows,
                block);

        var result =
            new List<RtdetrShadowRetryPass>(
                windows.Count);

        foreach (var window in windows)
        {
            token.ThrowIfCancellationRequested();

            using var cropView =
                new Mat(
                    source,
                    window.CropBounds);

            using var crop =
                cropView.Clone();

            var regions =
                AnalyzeMat(
                    crop,
                    new Point(
                        window.CropBounds.X,
                        window.CropBounds.Y),
                    $"ogkalu-rtdetr-v2-shadow-{window.PassId}",
                    $"SH_{window.PassId}_",
                    token);

            result.Add(
                new RtdetrShadowRetryPass(
                    window,
                    regions));
        }

        return result;
    }

    public static IReadOnlyList<RtdetrShadowRetryWindow> BuildShadowRetryWindows(
        int imageWidth,
        int imageHeight,
        OcrTextBlock block)
    {
        if (imageWidth <= 0 ||
            imageHeight <= 0)
        {
            return [];
        }

        var result =
            new List<RtdetrShadowRetryWindow>();

        foreach (var spec in
                 new[]
                 {
                     (PassId: "local_2p40", Context: 2.40),
                     (PassId: "local_1p60", Context: 1.60)
                 })
        {
            double blockMax =
                Math.Max(
                    1.0,
                    Math.Max(
                        block.W,
                        block.H));

            int desiredSide =
                (int)Math.Ceiling(
                    Math.Max(
                        256.0,
                        blockMax *
                        spec.Context));

            int side =
                Math.Clamp(
                    desiredSide,
                    64,
                    Math.Max(
                        64,
                        Math.Min(
                            imageWidth,
                            imageHeight)));

            double cx =
                block.X +
                block.W / 2.0;

            double cy =
                block.Y +
                block.H / 2.0;

            int left =
                (int)Math.Round(
                    cx -
                    side / 2.0);

            int top =
                (int)Math.Round(
                    cy -
                    side / 2.0);

            left =
                Math.Clamp(
                    left,
                    0,
                    Math.Max(
                        0,
                        imageWidth -
                        side));

            top =
                Math.Clamp(
                    top,
                    0,
                    Math.Max(
                        0,
                        imageHeight -
                        side));

            var crop =
                new Rect(
                    left,
                    top,
                    Math.Min(
                        side,
                        imageWidth -
                        left),
                    Math.Min(
                        side,
                        imageHeight -
                        top));

            if (crop.Width <= 0 ||
                crop.Height <= 0)
            {
                continue;
            }

            if (result.Any(x =>
                    x.CropBounds ==
                    crop))
            {
                continue;
            }

            result.Add(
                new RtdetrShadowRetryWindow(
                    spec.PassId,
                    crop,
                    spec.Context,
                    InputSize /
                    (double)Math.Max(
                        crop.Width,
                        crop.Height)));
        }

        return result;
    }

    public static bool IsShadowRetryMatch(
        OcrTextBlock block,
        PageRegion region)
    {
        if (region.Kind !=
            PageRegionKind.TextBubble)
        {
            return false;
        }

        var blockBounds =
            new Rect2d(
                block.X,
                block.Y,
                Math.Max(
                    1.0,
                    block.W),
                Math.Max(
                    1.0,
                    block.H));

        var target =
            region.Bounds;

        double left =
            Math.Max(
                blockBounds.Left,
                target.Left);

        double top =
            Math.Max(
                blockBounds.Top,
                target.Top);

        double right =
            Math.Min(
                blockBounds.Right,
                target.Right);

        double bottom =
            Math.Min(
                blockBounds.Bottom,
                target.Bottom);

        double intersection =
            Math.Max(
                0,
                right -
                left) *
            Math.Max(
                0,
                bottom -
                top);

        if (intersection <= 0)
            return false;

        double smaller =
            Math.Max(
                1.0,
                Math.Min(
                    blockBounds.Width *
                    blockBounds.Height,
                    target.Width *
                    (double)target.Height));

        double overlap =
            intersection /
            smaller;

        bool centerRelated =
            ContainsCenter(
                blockBounds,
                target) ||
            ContainsCenter(
                target,
                blockBounds);

        return overlap >=
                   0.25 ||
               centerRelated;
    }

    public static double ShadowRetryAffinity(
        OcrTextBlock block,
        PageRegion region)
    {
        if (!IsShadowRetryMatch(
                block,
                region))
        {
            return 0;
        }

        var blockBounds =
            new Rect2d(
                block.X,
                block.Y,
                Math.Max(
                    1.0,
                    block.W),
                Math.Max(
                    1.0,
                    block.H));

        double left =
            Math.Max(
                blockBounds.Left,
                region.Bounds.Left);

        double top =
            Math.Max(
                blockBounds.Top,
                region.Bounds.Top);

        double right =
            Math.Min(
                blockBounds.Right,
                region.Bounds.Right);

        double bottom =
            Math.Min(
                blockBounds.Bottom,
                region.Bounds.Bottom);

        double intersection =
            Math.Max(
                0,
                right -
                left) *
            Math.Max(
                0,
                bottom -
                top);

        double smaller =
            Math.Max(
                1.0,
                Math.Min(
                    blockBounds.Width *
                    blockBounds.Height,
                    region.Bounds.Width *
                    (double)region.Bounds.Height));

        return
            intersection /
            smaller *
            10.0 +
            region.Score;
    }

    static IReadOnlyList<PageRegion> AnalyzeMat(
        Mat source,
        Point sourceOrigin,
        string sourceTag,
        string idPrefix,
        CancellationToken token)
    {
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
                score <
                DefaultThreshold)
            {
                continue;
            }

            var kind =
                labels[i] switch
                {
                    0 =>
                        PageRegionKind.Bubble,
                    1 =>
                        PageRegionKind.TextBubble,
                    2 =>
                        PageRegionKind.TextFree,
                    _ =>
                        PageRegionKind.Unknown
                };

            if (kind ==
                PageRegionKind.Unknown)
            {
                continue;
            }

            int offset =
                i *
                4;

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

            var localBounds =
                ClampRect(
                    left,
                    top,
                    right -
                    left,
                    bottom -
                    top,
                    source.Cols,
                    source.Rows);

            if (localBounds.Width <
                    4 ||
                localBounds.Height <
                    4)
            {
                continue;
            }

            var sourceBounds =
                new Rect(
                    localBounds.X +
                    sourceOrigin.X,
                    localBounds.Y +
                    sourceOrigin.Y,
                    localBounds.Width,
                    localBounds.Height);

            raw.Add(
                new PageRegion(
                    "",
                    kind,
                    sourceBounds,
                    score,
                    sourceTag));
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
                            $"{idPrefix}{index + 1:000}"
                    })
            .ToList();
    }

    static bool ContainsCenter(
        Rect2d inner,
        Rect outer)
    {
        double cx =
            inner.X +
            inner.Width / 2.0;

        double cy =
            inner.Y +
            inner.Height / 2.0;

        return cx >=
                   outer.Left &&
               cx <=
                   outer.Right &&
               cy >=
                   outer.Top &&
               cy <=
                   outer.Bottom;
    }

    static bool ContainsCenter(
        Rect inner,
        Rect2d outer)
    {
        double cx =
            inner.X +
            inner.Width / 2.0;

        double cy =
            inner.Y +
            inner.Height / 2.0;

        return cx >=
                   outer.Left &&
               cx <=
                   outer.Right &&
               cy >=
                   outer.Top &&
               cy <=
                   outer.Bottom;
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
