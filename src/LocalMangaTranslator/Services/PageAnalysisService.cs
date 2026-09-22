using System.Text.Json;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Architecture 2.0 page analysis branch.
/// Region detection happens before and independently from OCR recognition.
/// RT-DETR owns the page region map. The legacy geometric detector may only
/// supply a better interior mask for an already detected RT-DETR bubble; it
/// cannot create additional containers in Architecture 2.0 mode.
/// </summary>
public sealed class PageAnalysisService : IDisposable
{
    readonly PageContainerCandidateDetector legacyDetector = new();
    readonly RtdetrPageRegionAnalyzer rtdetr = new();

    public Task<PageAnalysisResult> AnalyzeAsync(
        string sourcePath,
        PipelineOptions options,
        CancellationToken token = default)
        => Task.Run(
            () => Analyze(
                sourcePath,
                options,
                token),
            token);

    PageAnalysisResult Analyze(
        string sourcePath,
        PipelineOptions options,
        CancellationToken token)
    {
        token.ThrowIfCancellationRequested();

        var legacy =
            legacyDetector.Detect(
                sourcePath,
                token)
            .ToList();

        if (options.RegionAnalysis ==
                RegionAnalysisMode.Legacy ||
            !rtdetr.IsReady)
        {
            return new PageAnalysisResult(
                [],
                legacy,
                "legacy",
                false,
                options.RegionAnalysis ==
                    RegionAnalysisMode.HybridRtdetr
                    ? "rtdetr_model_missing"
                    : null);
        }

        var regions =
            rtdetr.Analyze(
                sourcePath,
                token)
            .ToList();

        var fused =
            FuseContainers(
                legacy,
                regions,
                sourcePath,
                token);

        return new PageAnalysisResult(
            regions,
            fused,
            "hybrid_rtdetr",
            true,
            $"regions={regions.Count}; legacy_mask_candidates={legacy.Count}; active_region_containers={fused.Count}");
    }

    public async Task WriteShadowDetectorRetryAuditAsync(
        string sourcePath,
        string outputDirectory,
        IReadOnlyList<LayaDetectorRetryRequest> requests,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (requests.Count == 0 ||
            !rtdetr.IsReady)
        {
            return;
        }

        await Task.Run(
            () =>
            {
                var requestAudits =
                    new List<object>();

                int recovered =
                    0;

                foreach (var request in requests)
                {
                    token.ThrowIfCancellationRequested();

                    progress?.Report(
                        $"Laya detector Shadow retry · id={request.Region.Id}");

                    var passes =
                        rtdetr.AnalyzeShadowRetry(
                            sourcePath,
                            request.Region.Source,
                            token);

                    var passAudits =
                        passes.Select(pass =>
                        {
                            var textBubbles =
                                pass.Regions
                                    .Where(x =>
                                        x.Kind ==
                                            PageRegionKind.TextBubble)
                                    .ToList();

                            var matches =
                                textBubbles
                                    .Where(x =>
                                        RtdetrPageRegionAnalyzer
                                            .IsShadowRetryMatch(
                                                request.Region.Source,
                                                x))
                                    .OrderByDescending(x =>
                                        RtdetrPageRegionAnalyzer
                                            .ShadowRetryAffinity(
                                                request.Region.Source,
                                                x))
                                    .ToList();

                            var best =
                                matches.FirstOrDefault();

                            return new
                            {
                                pass.Window.PassId,
                                CropBounds =
                                    new
                                    {
                                        pass.Window.CropBounds.X,
                                        pass.Window.CropBounds.Y,
                                        pass.Window.CropBounds.Width,
                                        pass.Window.CropBounds.Height
                                    },
                                pass.Window.ContextMultiplier,
                                pass.Window.InputPixelsPerSourcePixel,
                                RegionCount =
                                    pass.Regions.Count,
                                TextBubbleCount =
                                    textBubbles.Count,
                                Recovered =
                                    best is not null,
                                BestMatch =
                                    best is null
                                        ? null
                                        : new
                                        {
                                            best.RegionId,
                                            Kind =
                                                best.Kind.ToString(),
                                            Bounds =
                                                new
                                                {
                                                    best.Bounds.X,
                                                    best.Bounds.Y,
                                                    best.Bounds.Width,
                                                    best.Bounds.Height
                                                },
                                            best.Score,
                                            best.Source,
                                            Affinity =
                                                RtdetrPageRegionAnalyzer
                                                    .ShadowRetryAffinity(
                                                        request.Region.Source,
                                                        best)
                                        },
                                Matches =
                                    matches.Select(x =>
                                        new
                                        {
                                            x.RegionId,
                                            Bounds =
                                                new
                                                {
                                                    x.Bounds.X,
                                                    x.Bounds.Y,
                                                    x.Bounds.Width,
                                                    x.Bounds.Height
                                                },
                                            x.Score,
                                            Affinity =
                                                RtdetrPageRegionAnalyzer
                                                    .ShadowRetryAffinity(
                                                        request.Region.Source,
                                                        x)
                                        })
                                    .ToArray()
                            };
                        })
                        .ToArray();

                    bool requestRecovered =
                        passAudits.Any(x =>
                            x.Recovered);

                    if (requestRecovered)
                        recovered++;

                    requestAudits.Add(
                        new
                        {
                            TranslationRegionId =
                                request.Region.Id,
                            SourceText =
                                request.Region.Source.Text,
                            CorrectedText =
                                request.Region.CorrectedText,
                            LayaConfidence =
                                request.Confidence,
                            Block =
                                new
                                {
                                    request.Region.Source.X,
                                    request.Region.Source.Y,
                                    request.Region.Source.W,
                                    request.Region.Source.H
                                },
                            Recovered =
                                requestRecovered,
                            Passes =
                                passAudits
                        });
                }

                string page =
                    Path.GetFileNameWithoutExtension(
                        sourcePath);

                string path =
                    Path.Combine(
                        OutputDirectoryLayout.Debug(
                            outputDirectory),
                        page +
                        ".laya_detector_retry.json");

                var document =
                    new
                    {
                        Schema =
                            "pipeline-v2-laya-detector-retry-shadow-v1",
                        SourceFile =
                            Path.GetFileName(
                                sourcePath),
                        Mode =
                            "Shadow",
                        MutationApplied =
                            false,
                        Requests =
                            requests.Count,
                        RecoveredRequests =
                            recovered,
                        UnrecoveredRequests =
                            requests.Count -
                            recovered,
                        Items =
                            requestAudits
                    };

                File.WriteAllText(
                    path,
                    JsonSerializer.Serialize(
                        document,
                        new JsonSerializerOptions
                        {
                            WriteIndented =
                                true
                        }));

                progress?.Report(
                    $"Laya detector Shadow retry 완료 · {recovered}/{requests.Count} 재검출");
            },
            token);
    }

    public static IReadOnlyList<ContainerCandidate> FuseContainers(
        IReadOnlyList<ContainerCandidate> legacy,
        IReadOnlyList<PageRegion> regions,
        string sourcePath,
        CancellationToken token = default)
    {
        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            throw new InvalidOperationException(
                "페이지 분석을 위해 원본 이미지를 열 수 없습니다.");

        // 0011 region-first rule:
        // only RT-DETR Bubble regions may become speech containers.
        // Legacy candidates are mask donors only and can never introduce an
        // additional container by themselves.
        var bubbles =
            regions
                .Where(x =>
                    x.Kind == PageRegionKind.Bubble &&
                    x.Score >= 0.30f)
                .OrderBy(x => x.Bounds.Y)
                .ThenBy(x => x.Bounds.X)
                .ToList();

        var active =
            new List<ContainerCandidate>(
                bubbles.Count);

        var usedLegacy =
            new HashSet<string>(
                StringComparer.Ordinal);

        foreach (var bubble in bubbles)
        {
            token.ThrowIfCancellationRequested();

            var bounds =
                ClampRect(
                    bubble.Bounds,
                    source.Cols,
                    source.Rows);

            if (bounds.Width < 24 ||
                bounds.Height < 18)
            {
                continue;
            }

            var bestLegacy =
                legacy
                    .Where(candidate =>
                        !usedLegacy.Contains(
                            candidate.CandidateId))
                    .Select(candidate => new
                    {
                        Candidate = candidate,
                        IoU = IoU(
                            candidate.Bounds,
                            bounds),
                        BubbleCoverage =
                            IntersectionArea(
                                candidate.Bounds,
                                bounds) /
                            Math.Max(
                                1.0,
                                bounds.Width *
                                (double)bounds.Height),
                        CandidateCoverage =
                            IntersectionArea(
                                candidate.Bounds,
                                bounds) /
                            Math.Max(
                                1.0,
                                candidate.Bounds.Width *
                                (double)candidate.Bounds.Height)
                    })
                    .Where(x =>
                        x.IoU >= 0.22 ||
                        (x.BubbleCoverage >= 0.45 &&
                         x.CandidateCoverage >= 0.45))
                    .OrderByDescending(x => x.IoU)
                    .ThenByDescending(x =>
                        Math.Min(
                            x.BubbleCoverage,
                            x.CandidateCoverage))
                    .ThenByDescending(x => x.Candidate.Score)
                    .FirstOrDefault();

            if (bestLegacy is not null)
            {
                usedLegacy.Add(
                    bestLegacy.Candidate.CandidateId);

                // Keep the contour-derived bounds/mask only because the learned
                // region already established that this physical area is a
                // speech bubble.
                active.Add(
                    bestLegacy.Candidate with
                    {
                        CandidateId = "",
                        Kind =
                            ContainerCandidateKind.Speech,
                        DetectorMode =
                            "rtdetr_region+legacy_mask",
                        Score =
                            Math.Max(
                                5.0,
                                bestLegacy.Candidate.Score) +
                            bubble.Score * 1.5,
                        RegionId =
                            bubble.RegionId,
                        LearnedBounds =
                            bounds
                    });

                continue;
            }

            var (mask, fillRatio) =
                BuildSafeRectangleMask(
                    bounds);

            active.Add(
                new ContainerCandidate(
                    "",
                    ContainerCandidateKind.Speech,
                    bounds,
                    "rtdetr_region_rect",
                    false,
                    5.0 +
                    bubble.Score * 2.0,
                    fillRatio,
                    CountBorderTouches(
                        bounds,
                        source.Cols,
                        source.Rows),
                    mask,
                    bounds.Width,
                    bounds.Height)
                {
                    RegionId =
                        bubble.RegionId,
                    LearnedBounds =
                        bounds
                });
        }

        var canonical =
            Canonicalize(
                active);

        return canonical
            .Select(
                (x, index) =>
                    x with
                    {
                        CandidateId =
                            $"PC{index + 1:000}"
                    })
            .ToList();
    }

    static double IoU(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        if (intersection <= 0)
            return 0;

        double areaA =
            Math.Max(
                1,
                a.Width *
                (double)a.Height);

        double areaB =
            Math.Max(
                1,
                b.Width *
                (double)b.Height);

        return intersection /
               Math.Max(
                   1,
                   areaA +
                   areaB -
                   intersection);
    }

    static (byte[] Mask, double FillRatio) BuildSafeRectangleMask(
        Rect bounds)
    {
        int width =
            Math.Max(
                1,
                bounds.Width);

        int height =
            Math.Max(
                1,
                bounds.Height);

        var mask =
            new byte[
                width *
                height];

        int guard =
            Math.Clamp(
                Math.Min(
                    width,
                    height) / 24,
                2,
                10);

        int left =
            guard;

        int top =
            guard;

        int right =
            Math.Max(
                left,
                width - guard);

        int bottom =
            Math.Max(
                top,
                height - guard);

        long count = 0;

        for (int y = top; y < bottom; y++)
        {
            for (int x = left; x < right; x++)
            {
                mask[y * width + x] =
                    255;

                count++;
            }
        }

        return (
            mask,
            count /
            (double)Math.Max(
                1,
                width * (long)height));
    }

    static List<ContainerCandidate> Canonicalize(
        IReadOnlyList<ContainerCandidate> input)
    {
        var ordered =
            input
                .OrderByDescending(
                    x => x.Score)
                .ThenByDescending(
                    x =>
                        x.Bounds.Width *
                        (long)x.Bounds.Height)
                .ToList();

        var kept =
            new List<ContainerCandidate>();

        foreach (var candidate in ordered)
        {
            int duplicate =
                kept.FindIndex(
                    existing =>
                        SamePhysicalArea(
                            existing.Bounds,
                            candidate.Bounds));

            if (duplicate < 0)
            {
                kept.Add(
                    candidate);

                continue;
            }

            // Prefer real contour masks over RT-DETR rectangle fallbacks when
            // the learned detector and geometric detector refer to the same
            // physical bubble.
            bool candidateHasContour =
                candidate.DetectorMode.Contains(
                    "+legacy_mask",
                    StringComparison.Ordinal);

            bool existingFallback =
                string.Equals(
                    kept[duplicate].DetectorMode,
                    "rtdetr_region_rect",
                    StringComparison.Ordinal);

            if (candidateHasContour &&
                existingFallback)
            {
                kept[duplicate] =
                    candidate;
            }
        }

        return kept
            .OrderBy(
                x => x.Bounds.Y)
            .ThenBy(
                x => x.Bounds.X)
            .Take(100)
            .ToList();
    }

    static bool SamePhysicalArea(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        if (intersection <= 0)
            return false;

        double areaA =
            Math.Max(
                1,
                a.Width *
                (double)a.Height);

        double areaB =
            Math.Max(
                1,
                b.Width *
                (double)b.Height);

        double containment =
            intersection /
            Math.Min(
                areaA,
                areaB);

        double iou =
            intersection /
            Math.Max(
                1,
                areaA +
                areaB -
                intersection);

        return containment >= 0.82 ||
               iou >= 0.58;
    }

    static double Containment(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        if (intersection <= 0)
            return 0;

        double smaller =
            Math.Max(
                1,
                Math.Min(
                    a.Width *
                    (double)a.Height,
                    b.Width *
                    (double)b.Height));

        return intersection /
               smaller;
    }

    static double IntersectionArea(
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

        return Math.Max(
                   0,
                   right - left) *
               (double)Math.Max(
                   0,
                   bottom - top);
    }

    static Rect ClampRect(
        Rect rect,
        int imageWidth,
        int imageHeight)
    {
        int left =
            Math.Clamp(
                rect.Left,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                rect.Top,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                rect.Right,
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                rect.Bottom,
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

    static int CountBorderTouches(
        Rect bounds,
        int imageWidth,
        int imageHeight)
    {
        const int tolerance =
            3;

        int touches =
            0;

        if (bounds.Left <= tolerance)
            touches++;

        if (bounds.Top <= tolerance)
            touches++;

        if (bounds.Right >=
            imageWidth - tolerance)
        {
            touches++;
        }

        if (bounds.Bottom >=
            imageHeight - tolerance)
        {
            touches++;
        }

        return touches;
    }

    public void Dispose()
        => rtdetr.Dispose();
}
