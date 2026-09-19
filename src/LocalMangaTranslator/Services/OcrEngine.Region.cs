using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed partial class OcrEngine
{
    public Task<RegionOcrBatch> RecognizeRegionsAsync(
        string imagePath,
        IReadOnlyList<PageRegion> regions,
        IProgress<PipelineProgress>? progress,
        CancellationToken token)
        => Task.Run(
            () => RecognizeRegions(
                imagePath,
                regions,
                progress,
                token),
            token);

    RegionOcrBatch RecognizeRegions(
        string imagePath,
        IReadOnlyList<PageRegion> regions,
        IProgress<PipelineProgress>? progress,
        CancellationToken token)
    {
        var observations =
            new List<OcrObservation>();

        var attempts =
            new List<RegionOcrAttempt>();

        var textRegions =
            regions
                .Where(
                    x =>
                        x.Kind is
                            PageRegionKind.TextBubble or
                            PageRegionKind.TextFree)
                .Where(
                    x =>
                        x.Score >= 0.25f)
                .OrderBy(
                    x => x.Bounds.Y)
                .ThenBy(
                    x => x.Bounds.X)
                .ToList();

        if (textRegions.Count == 0)
            return new(
                observations,
                attempts);

        using var input =
            File.OpenRead(
                imagePath);

        var frame =
            BitmapDecoder.Create(
                input,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad)
            .Frames[0];

        foreach (var region in textRegions)
        {
            token.ThrowIfCancellationRequested();

            var cropBounds =
                ExpandAndClamp(
                    region.Bounds,
                    frame.PixelWidth,
                    frame.PixelHeight);

            foreach (var (pass, scale) in
                     new[]
                     {
                         (OcrPassKind.Region1x, 1.0),
                         (OcrPassKind.Region2x, 2.0)
                     })
            {
                token.ThrowIfCancellationRequested();

                if (!Ready ||
                    cropBounds.Width < 4 ||
                    cropBounds.Height < 4)
                {
                    attempts.Add(
                        new RegionOcrAttempt(
                            region.RegionId,
                            pass,
                            "skipped",
                            0,
                            !Ready
                                ? "ocr_not_ready"
                                : "invalid_crop"));

                    continue;
                }

                progress?.Report(
                    new PipelineProgress(
                        PipelineStageKind.RegionOcr,
                        $"독립 텍스트 영역 OCR · {region.RegionId} · {scale:0}배"));

                string temporary =
                    Path.Combine(
                        Path.GetTempPath(),
                        $"lmt_region_{Guid.NewGuid():N}.png");

                try
                {
                    var crop =
                        new CroppedBitmap(
                            frame,
                            new Int32Rect(
                                cropBounds.X,
                                cropBounds.Y,
                                cropBounds.Width,
                                cropBounds.Height));

                    var transformed =
                        new TransformedBitmap(
                            crop,
                            new ScaleTransform(
                                scale,
                                scale));

                    var encoder =
                        new PngBitmapEncoder();

                    encoder.Frames.Add(
                        BitmapFrame.Create(
                            transformed));

                    using (var output =
                           File.Create(
                               temporary))
                    {
                        encoder.Save(
                            output);
                    }

                    token.ThrowIfCancellationRequested();

                    var detected =
                        DetectToObservations(
                            temporary,
                            scale,
                            pass,
                            region.RegionId,
                            cropBounds.X,
                            cropBounds.Y);

                    observations.AddRange(
                        detected);

                    attempts.Add(
                        new RegionOcrAttempt(
                            region.RegionId,
                            pass,
                            "completed",
                            detected.Count));
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    attempts.Add(
                        new RegionOcrAttempt(
                            region.RegionId,
                            pass,
                            "failed",
                            0,
                            ex.Message));
                }
                finally
                {
                    TryDelete(
                        temporary);
                }
            }
        }

        return new RegionOcrBatch(
            observations
                .Select(
                    (o, index) =>
                        o with
                        {
                            ObservationId =
                                $"RO{index + 1:0000}"
                        })
                .ToList(),
            attempts);
    }

    static Int32Rect ExpandAndClamp(
        OpenCvSharp.Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        int pad =
            Math.Clamp(
                Math.Min(
                    bounds.Width,
                    bounds.Height) / 12,
                3,
                20);

        int left =
            Math.Clamp(
                bounds.Left - pad,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                bounds.Top - pad,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                bounds.Right + pad,
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                bounds.Bottom + pad,
                0,
                imageHeight);

        return new Int32Rect(
            left,
            top,
            Math.Max(
                0,
                right - left),
            Math.Max(
                0,
                bottom - top));
    }
}
