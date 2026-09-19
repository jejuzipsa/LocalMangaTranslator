using System.Globalization;
using System.Text.Json;
using LocalMangaTranslator.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// C# ONNX Runtime port of the Apache-2.0 Baberu OCR reference runner.
/// Baberu reads one speech-bubble crop and returns text; page detection remains
/// the responsibility of RT-DETR.
/// </summary>
public sealed class BaberuOcrEngine : IDisposable
{
    const int InputSize = 224;
    const int MaxNewTokens = 128;
    const double RepetitionPenalty = 1.2;
    const int MaxContentRun = 12;

    static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    static readonly string[] PastNames =
    [
        "past_k0", "past_k1", "past_k2", "past_k3", "past_k4", "past_k5",
        "past_v0", "past_v1", "past_v2", "past_v3", "past_v4", "past_v5"
    ];

    readonly object sync = new();

    InferenceSession? vision;
    InferenceSession? prefill;
    InferenceSession? step;

    Dictionary<int, string>? idToChar;
    HashSet<int>? contentIds;

    public bool IsReady =>
        ExternalModelManager.IsBaberuReady();

    public IReadOnlyList<SecondaryOcrEvidence> Analyze(
        string sourcePath,
        PageAnalysisResult pageAnalysis,
        CancellationToken token = default)
    {
        if (!IsReady)
            return [];

        token.ThrowIfCancellationRequested();

        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "Baberu OCR을 위해 원본 이미지를 열 수 없습니다.");

        EnsureLoaded();

        var bubbles =
            pageAnalysis.Regions
                .Where(x =>
                    x.Kind == PageRegionKind.Bubble &&
                    x.Score >= 0.30f)
                .OrderBy(x => x.Bounds.Y)
                .ThenBy(x => x.Bounds.X)
                .ToList();

        var textRegions =
            pageAnalysis.Regions
                .Where(x =>
                    x.Kind == PageRegionKind.TextBubble &&
                    x.Score >= 0.24f)
                .ToList();

        var output =
            new List<SecondaryOcrEvidence>();

        int next = 1;

        foreach (var bubble in bubbles)
        {
            token.ThrowIfCancellationRequested();

            var relatedText =
                textRegions
                    .Where(x =>
                        CenterInside(
                            x.Bounds,
                            bubble.Bounds) ||
                        Coverage(
                            x.Bounds,
                            bubble.Bounds) >= 0.45)
                    .OrderByDescending(x => x.Score)
                    .ToList();

            // Baberu is a bubble OCR, but we still require learned text-region
            // evidence before its output can participate in the canonical text
            // path. This prevents "empty bubble -> hallucinated text".
            if (relatedText.Count == 0)
            {
                output.Add(
                    new SecondaryOcrEvidence(
                        $"BO{next++:0000}",
                        bubble.RegionId,
                        bubble.Bounds,
                        "",
                        "baberu",
                        false,
                        "no_text_region"));

                continue;
            }

            var cropBounds =
                ExpandAndClamp(
                    bubble.Bounds,
                    source.Cols,
                    source.Rows);

            using var crop =
                new Mat(
                    source,
                    cropBounds)
                .Clone();

            string text =
                Recognize(
                    crop,
                    token)
                .Trim();

            var evidenceBounds =
                UnionBounds(
                    relatedText.Select(x => x.Bounds),
                    source.Cols,
                    source.Rows);

            bool accepted =
                MeaningfulLength(text) >= 1;

            output.Add(
                new SecondaryOcrEvidence(
                    $"BO{next++:0000}",
                    bubble.RegionId,
                    evidenceBounds,
                    text,
                    "baberu",
                    accepted,
                    accepted
                        ? "bubble_text_region"
                        : "empty_result"));
        }

        return output;
    }

    string Recognize(
        Mat crop,
        CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        EnsureLoaded();

        var pixels =
            Preprocess(
                crop);

        var imageTensor =
            new DenseTensor<float>(
                pixels,
                new[]
                {
                    1,
                    3,
                    InputSize,
                    InputSize
                });

        using var visionResults =
            vision!.Run(
                [
                    NamedOnnxValue.CreateFromTensor(
                        "pixel_values",
                        imageTensor)
                ]);

        var visionOutput =
            visionResults
                .FirstOrDefault(x =>
                    string.Equals(
                        x.Name,
                        "vision_embeds",
                        StringComparison.OrdinalIgnoreCase))
                ?? visionResults.First();

        var visionTensor =
            CopyTensor(
                visionOutput.AsTensor<float>());

        var bos =
            new DenseTensor<long>(
                new long[] { 1 },
                new[] { 1, 1 });

        using var prefillResults =
            prefill!.Run(
                [
                    NamedOnnxValue.CreateFromTensor(
                        "vision_embeds",
                        visionTensor),
                    NamedOnnxValue.CreateFromTensor(
                        "input_ids",
                        bos)
                ]);

        var prefillList =
            prefillResults.ToList();

        double[] logits =
            LastLogits(
                prefillList[0].AsTensor<float>());

        var present =
            prefillList
                .Skip(1)
                .Select(x =>
                    CopyTensor(
                        x.AsTensor<float>()))
                .ToList();

        var sequence =
            new List<int>
            {
                1
            };

        var generated =
            new List<int>();

        int visionTokens =
            visionTensor.Dimensions.Count >= 2
                ? visionTensor.Dimensions[1]
                : 256;

        long position =
            visionTokens + 1;

        for (int generatedCount = 0;
             generatedCount < MaxNewTokens;
             generatedCount++)
        {
            token.ThrowIfCancellationRequested();

            ApplyRepetitionPenalty(
                logits,
                sequence);

            ApplyContentRunCap(
                logits,
                generated);

            int nextToken =
                ArgMax(
                    logits);

            if (nextToken == 2)
                break;

            generated.Add(
                nextToken);

            sequence.Add(
                nextToken);

            var inputIds =
                new DenseTensor<long>(
                    new long[]
                    {
                        nextToken
                    },
                    new[]
                    {
                        1,
                        1
                    });

            var positionIds =
                new DenseTensor<long>(
                    new long[]
                    {
                        position
                    },
                    new[]
                    {
                        1,
                        1
                    });

            var inputs =
                new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(
                        "input_ids",
                        inputIds),
                    NamedOnnxValue.CreateFromTensor(
                        "position_ids",
                        positionIds)
                };

            for (int i = 0;
                 i < Math.Min(
                     PastNames.Length,
                     present.Count);
                 i++)
            {
                inputs.Add(
                    NamedOnnxValue.CreateFromTensor(
                        PastNames[i],
                        present[i]));
            }

            using var stepResults =
                step!.Run(
                    inputs);

            var stepList =
                stepResults.ToList();

            logits =
                LastLogits(
                    stepList[0].AsTensor<float>());

            present =
                stepList
                    .Skip(1)
                    .Select(x =>
                        CopyTensor(
                            x.AsTensor<float>()))
                    .ToList();

            position++;
        }

        return Decode(
            generated);
    }

    void EnsureLoaded()
    {
        lock (sync)
        {
            if (vision is not null &&
                prefill is not null &&
                step is not null &&
                idToChar is not null &&
                contentIds is not null)
            {
                return;
            }

            if (!IsReady)
            {
                throw new InvalidOperationException(
                    "Baberu OCR 모델이 설치되어 있지 않습니다.");
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

            vision =
                new InferenceSession(
                    ExternalModelManager.BaberuVisionPath,
                    options);

            prefill =
                new InferenceSession(
                    ExternalModelManager.BaberuPrefillPath,
                    options);

            step =
                new InferenceSession(
                    ExternalModelManager.BaberuStepPath,
                    options);

            var charset =
                JsonSerializer.Deserialize<List<string>>(
                    File.ReadAllText(
                        ExternalModelManager.BaberuVocabPath))
                ?? [];

            idToChar =
                new Dictionary<int, string>
                {
                    [0] = "",
                    [1] = "",
                    [2] = "",
                    [3] = ""
                };

            contentIds =
                [];

            for (int i = 0;
                 i < charset.Count;
                 i++)
            {
                int id =
                    i + 4;

                string ch =
                    charset[i];

                idToChar[id] =
                    ch;

                if (IsContentCharacter(
                        ch))
                {
                    contentIds.Add(
                        id);
                }
            }
        }
    }

    static float[] Preprocess(
        Mat crop)
    {
        using var resized =
            new Mat();

        Cv2.Resize(
            crop,
            resized,
            new Size(
                InputSize,
                InputSize),
            0,
            0,
            InterpolationFlags.Cubic);

        using var rgb =
            new Mat();

        Cv2.CvtColor(
            resized,
            rgb,
            ColorConversionCodes.BGR2RGB);

        var output =
            new float[
                3 *
                InputSize *
                InputSize];

        int plane =
            InputSize *
            InputSize;

        for (int y = 0;
             y < InputSize;
             y++)
        {
            for (int x = 0;
                 x < InputSize;
                 x++)
            {
                var px =
                    rgb.At<Vec3b>(
                        y,
                        x);

                for (int channel = 0;
                     channel < 3;
                     channel++)
                {
                    output[
                        channel * plane +
                        y * InputSize +
                        x] =
                        (px[channel] / 255f -
                         Mean[channel]) /
                        Std[channel];
                }
            }
        }

        return output;
    }

    static DenseTensor<float> CopyTensor(
        Tensor<float> tensor)
        => new(
            tensor.ToArray(),
            tensor.Dimensions.ToArray());

    static double[] LastLogits(
        Tensor<float> tensor)
    {
        var dims =
            tensor.Dimensions.ToArray();

        if (dims.Length == 0)
            return [];

        int vocab =
            dims[^1];

        if (vocab <= 0)
            return [];

        var data =
            tensor.ToArray();

        int offset =
            Math.Max(
                0,
                data.Length -
                vocab);

        var output =
            new double[vocab];

        for (int i = 0;
             i < vocab;
             i++)
        {
            output[i] =
                data[offset + i];
        }

        return output;
    }

    static void ApplyRepetitionPenalty(
        double[] logits,
        IReadOnlyList<int> sequence)
    {
        foreach (int token in
                 sequence.Distinct())
        {
            if (token < 0 ||
                token >= logits.Length)
            {
                continue;
            }

            double score =
                logits[token];

            logits[token] =
                score < 0
                    ? score *
                      RepetitionPenalty
                    : score /
                      RepetitionPenalty;
        }
    }

    void ApplyContentRunCap(
        double[] logits,
        IReadOnlyList<int> generated)
    {
        if (contentIds is null ||
            generated.Count == 0)
        {
            return;
        }

        int last =
            generated[^1];

        if (!contentIds.Contains(
                last))
        {
            return;
        }

        int run =
            0;

        for (int i =
                 generated.Count - 1;
             i >= 0;
             i--)
        {
            if (generated[i] != last)
                break;

            run++;
        }

        if (run >=
                MaxContentRun &&
            last >= 0 &&
            last < logits.Length)
        {
            logits[last] =
                double.NegativeInfinity;
        }
    }

    static int ArgMax(
        IReadOnlyList<double> values)
    {
        if (values.Count == 0)
            return 2;

        int index =
            0;

        double best =
            values[0];

        for (int i = 1;
             i < values.Count;
             i++)
        {
            if (values[i] <= best)
                continue;

            best =
                values[i];

            index =
                i;
        }

        return index;
    }

    string Decode(
        IEnumerable<int> ids)
    {
        if (idToChar is null)
            return "";

        return string.Concat(
            ids
                .Where(x => x >= 4)
                .Select(x =>
                    idToChar.TryGetValue(
                        x,
                        out var ch)
                        ? ch
                        : ""));
    }

    static bool IsContentCharacter(
        string value)
    {
        if (value.Length != 1 ||
            value is "ー" or "ｰ" or "〜" or "~")
        {
            return false;
        }

        UnicodeCategory category =
            char.GetUnicodeCategory(
                value[0]);

        return category is
            UnicodeCategory.UppercaseLetter or
            UnicodeCategory.LowercaseLetter or
            UnicodeCategory.TitlecaseLetter or
            UnicodeCategory.ModifierLetter or
            UnicodeCategory.OtherLetter or
            UnicodeCategory.DecimalDigitNumber or
            UnicodeCategory.LetterNumber or
            UnicodeCategory.OtherNumber;
    }

    static Rect ExpandAndClamp(
        Rect rect,
        int imageWidth,
        int imageHeight)
    {
        int pad =
            Math.Clamp(
                Math.Min(
                    rect.Width,
                    rect.Height) / 20,
                2,
                12);

        int left =
            Math.Clamp(
                rect.Left - pad,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                rect.Top - pad,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                rect.Right + pad,
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                rect.Bottom + pad,
                0,
                imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(
                1,
                right - left),
            Math.Max(
                1,
                bottom - top));
    }

    static Rect UnionBounds(
        IEnumerable<Rect> rects,
        int imageWidth,
        int imageHeight)
    {
        var list =
            rects.ToList();

        if (list.Count == 0)
        {
            return new Rect(
                0,
                0,
                1,
                1);
        }

        int left =
            list.Min(x => x.Left);

        int top =
            list.Min(x => x.Top);

        int right =
            list.Max(x => x.Right);

        int bottom =
            list.Max(x => x.Bottom);

        return new Rect(
            Math.Clamp(
                left,
                0,
                imageWidth),
            Math.Clamp(
                top,
                0,
                imageHeight),
            Math.Clamp(
                right,
                0,
                imageWidth) -
            Math.Clamp(
                left,
                0,
                imageWidth),
            Math.Clamp(
                bottom,
                0,
                imageHeight) -
            Math.Clamp(
                top,
                0,
                imageHeight));
    }

    static bool CenterInside(
        Rect inner,
        Rect outer)
    {
        double x =
            inner.X +
            inner.Width / 2.0;

        double y =
            inner.Y +
            inner.Height / 2.0;

        return x >= outer.Left &&
               x <= outer.Right &&
               y >= outer.Top &&
               y <= outer.Bottom;
    }

    static double Coverage(
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

        double area =
            Math.Max(
                1,
                a.Width *
                (double)a.Height);

        return intersection /
               area;
    }

    static int MeaningfulLength(
        string? text)
        => string.IsNullOrWhiteSpace(
               text)
            ? 0
            : text.Count(
                char.IsLetterOrDigit);

    public void Dispose()
    {
        lock (sync)
        {
            vision?.Dispose();
            prefill?.Dispose();
            step?.Dispose();

            vision = null;
            prefill = null;
            step = null;

            idToChar = null;
            contentIds = null;
        }
    }
}
