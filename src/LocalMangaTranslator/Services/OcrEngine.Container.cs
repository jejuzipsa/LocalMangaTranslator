using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed partial class OcrEngine
{
    public Task<ContainerOcrBatch> RecognizeContainersAsync(
        string imagePath,
        IReadOnlyList<ContainerCandidate> candidates,
        IProgress<PipelineProgress>? progress,
        CancellationToken token)
        => Task.Run(() => RecognizeContainers(imagePath, candidates, progress, token), token);

    ContainerOcrBatch RecognizeContainers(
        string imagePath,
        IReadOnlyList<ContainerCandidate> candidates,
        IProgress<PipelineProgress>? progress,
        CancellationToken token)
    {
        var observations = new List<OcrObservation>();
        var attempts = new List<ContainerOcrAttempt>();
        if (candidates.Count == 0) return new(observations, attempts);

        using var input = File.OpenRead(imagePath);
        var frame = BitmapDecoder.Create(input, BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad).Frames[0];

        foreach (var candidate in candidates)
        {
            token.ThrowIfCancellationRequested();
            var b = candidate.Bounds;
            foreach (var (pass, scale) in new[]
                     { (OcrPassKind.Container1x, 1.0), (OcrPassKind.Container2x, 2.0) })
            {
                token.ThrowIfCancellationRequested();
                if (!Ready || !ContainerOcrValidator.ValidateCandidate(candidate).Eligible ||
                    (long)b.X + b.Width > frame.PixelWidth || (long)b.Y + b.Height > frame.PixelHeight)
                {
                    attempts.Add(new(candidate.CandidateId, pass, "skipped", 0,
                        !Ready ? "ocr_not_ready" : "invalid_crop"));
                    continue;
                }

                progress?.Report(new(PipelineStageKind.ContainerOcr,
                    $"컨테이너 OCR · {candidate.CandidateId} · {scale:0}배"));
                string temporary = Path.Combine(Path.GetTempPath(), $"lmt_container_{Guid.NewGuid():N}.png");
                try
                {
                    var crop = new CroppedBitmap(frame, new Int32Rect(b.X, b.Y, b.Width, b.Height));
                    var transformed = new TransformedBitmap(crop, new ScaleTransform(scale, scale));
                    var encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(transformed));
                    using (var output = File.Create(temporary)) encoder.Save(output);
                    token.ThrowIfCancellationRequested();
                    var detected = DetectToObservations(temporary, scale, pass, candidate.CandidateId, b.X, b.Y);
                    token.ThrowIfCancellationRequested();
                    observations.AddRange(detected);
                    attempts.Add(new(candidate.CandidateId, pass, "completed", detected.Count));
                }
                catch (OperationCanceledException) { throw; }
                catch (Exception ex)
                {
                    attempts.Add(new(candidate.CandidateId, pass, "failed", 0, ex.Message));
                }
                finally { TryDelete(temporary); }
            }
        }
        return new(observations.Select((o, i) => o with { ObservationId = $"CO{i + 1:0000}" }).ToList(), attempts);
    }
}
