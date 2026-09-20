using System.Text.Json;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.Services;
using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public sealed record ErasePipelineV2Result(
    int TargetCount,
    int MaskPixels,
    string DetectionDebugPath,
    string MaskDebugPath,
    string CleanedDebugPath,
    string DetectionJsonPath);

/// <summary>
/// Independent V2 erase probe.
///
/// Input is only the source image plus immutable RT-DETR geometry.
/// It intentionally does not consume OCR lines, Vision text, translations,
/// ownership decisions, canonical lines, legacy balloon masks, or render plans.
/// </summary>
public sealed class ErasePipelineV2
{
    const double InpaintRadius = 3.0;

    public ErasePipelineV2Result Run(
        string sourcePath,
        string outputDirectory,
        V2DetectionSnapshot snapshot,
        CancellationToken token = default)
    {
        string debugDir = OutputDirectoryLayout.Debug(outputDirectory);
        Directory.CreateDirectory(debugDir);

        string name = Path.GetFileNameWithoutExtension(sourcePath);

        string detectionPath =
            Path.Combine(debugDir, $"{name}.v2_00_detection.webp");

        string maskPath =
            Path.Combine(debugDir, $"{name}.v2_01_text_mask.webp");

        string cleanedPath =
            Path.Combine(debugDir, $"{name}.v2_02_cleaned.webp");

        string jsonPath =
            Path.Combine(debugDir, $"{name}.v2_detection.json");

        using var source =
            Cv2.ImRead(sourcePath, ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "Pipeline V2 erase를 위해 원본 이미지를 열 수 없습니다.");

        using var mask = Mat.Zeros(
            source.Rows,
            source.Cols,
            MatType.CV_8UC1).ToMat();

        foreach (var target in snapshot.TextTargets)
        {
            token.ThrowIfCancellationRequested();

            using var targetMask =
                ComicTranslateComponentMask.Build(
                    source,
                    target.TextBounds,
                    target.BubbleBounds);

            Cv2.BitwiseOr(mask, targetMask, mask);
        }

        SaveDetectionDebug(
            source,
            snapshot,
            detectionPath);

        SaveMaskDebug(
            source,
            mask,
            maskPath);

        int pixels = Cv2.CountNonZero(mask);

        if (pixels == 0)
        {
            Cv2.ImWrite(
                cleanedPath,
                source,
                [new ImageEncodingParam(ImwriteFlags.WebPQuality, 101)]);
        }
        else
        {
            using var cleaned = new Mat();

            Cv2.Inpaint(
                source,
                mask,
                cleaned,
                InpaintRadius,
                InpaintTypes.Telea);

            Cv2.ImWrite(
                cleanedPath,
                cleaned,
                [new ImageEncodingParam(ImwriteFlags.WebPQuality, 101)]);
        }

        var diagnostic = new
        {
            SourceFile = Path.GetFileName(sourcePath),
            snapshot.SourceMode,
            TargetCount = snapshot.TextTargets.Count,
            MaskPixels = pixels,
            Targets = snapshot.TextTargets.Select(x => new
            {
                x.TextRegionId,
                TextBounds = new
                {
                    x.TextBounds.X,
                    x.TextBounds.Y,
                    x.TextBounds.Width,
                    x.TextBounds.Height
                },
                x.TextScore,
                x.BubbleRegionId,
                BubbleBounds = x.BubbleBounds is Rect b
                    ? new
                    {
                        b.X,
                        b.Y,
                        b.Width,
                        b.Height
                    }
                    : null,
                x.BubbleScore
            })
        };

        File.WriteAllText(
            jsonPath,
            JsonSerializer.Serialize(
                diagnostic,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));

        return new ErasePipelineV2Result(
            snapshot.TextTargets.Count,
            pixels,
            detectionPath,
            maskPath,
            cleanedPath,
            jsonPath);
    }

    static void SaveDetectionDebug(
        Mat source,
        V2DetectionSnapshot snapshot,
        string path)
    {
        using var debug = source.Clone();

        foreach (var region in snapshot.RawRegions)
        {
            Scalar color = region.Kind switch
            {
                LocalMangaTranslator.Models.PageRegionKind.Bubble =>
                    new Scalar(0, 220, 220),
                LocalMangaTranslator.Models.PageRegionKind.TextBubble =>
                    new Scalar(255, 180, 0),
                _ =>
                    new Scalar(130, 130, 130)
            };

            int thickness =
                region.Kind is
                    LocalMangaTranslator.Models.PageRegionKind.Bubble or
                    LocalMangaTranslator.Models.PageRegionKind.TextBubble
                    ? 3
                    : 1;

            Cv2.Rectangle(
                debug,
                region.Bounds,
                color,
                thickness);
        }

        Cv2.ImWrite(
            path,
            debug,
            [new ImageEncodingParam(ImwriteFlags.WebPQuality, 101)]);
    }

    static void SaveMaskDebug(
        Mat source,
        Mat mask,
        string path)
    {
        using var debug = source.Clone();
        using var red = new Mat(
            source.Rows,
            source.Cols,
            MatType.CV_8UC3,
            new Scalar(0, 0, 255));

        red.CopyTo(debug, mask);

        Cv2.ImWrite(
            path,
            debug,
            [new ImageEncodingParam(ImwriteFlags.WebPQuality, 101)]);
    }
}
