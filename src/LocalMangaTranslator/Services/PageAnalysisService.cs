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
public sealed record LayaDetectorAdvisoryCandidate(
    int TranslationRegionId,
    PageRegion TextRegion,
    PageRegion BubbleRegion,
    double? LayaConfidence,
    double CrossScaleTextIoU,
    double CrossScaleBubbleIoU,
    string SourceText);

public sealed record LayaDetectorRetryAuditResult(
    IReadOnlyList<LayaDetectorAdvisoryCandidate> AdvisoryCandidates)
{
    public static LayaDetectorRetryAuditResult Empty { get; } =
        new([]);
}

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

    public async Task<LayaDetectorRetryAuditResult> RunDetectorRetryAuditAsync(
        string sourcePath,
        string outputDirectory,
        IReadOnlyList<LayaDetectorRetryRequest> requests,
        IReadOnlyList<PageRegion> existingRegions,
        LayaDecisionMode mode,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (requests.Count == 0 ||
            !rtdetr.IsReady)
        {
            return LayaDetectorRetryAuditResult.Empty;
        }

        return await Task.Run(
            () =>
            {
                var requestAudits =
                    new List<object>();

                var advisoryCandidates =
                    new List<LayaDetectorAdvisoryCandidate>();

                int recovered =
                    0;

                int advisoryEligible =
                    0;

                foreach (var request in requests)
                {
                    token.ThrowIfCancellationRequested();

                    progress?.Report(
                        $"Laya detector retry · id={request.Region.Id}");

                    var passes =
                        rtdetr.AnalyzeShadowRetry(
                            sourcePath,
                            request.Region.Source,
                            token);

                    var evaluations =
                        passes
                            .Select(pass =>
                            {
                                var textMatches =
                                    pass.Regions
                                        .Where(x =>
                                            x.Kind ==
                                                PageRegionKind.TextBubble &&
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

                                var bestText =
                                    textMatches.FirstOrDefault();

                                var bestBubble =
                                    bestText is null
                                        ? null
                                        : FindOwningBubble(
                                            bestText,
                                            pass.Regions);

                                return (
                                    Pass: pass,
                                    TextMatches: textMatches,
                                    BestText: bestText,
                                    BestBubble: bestBubble);
                            })
                            .ToList();

                    bool requestRecovered =
                        evaluations.Any(x =>
                            x.BestText is not null);

                    if (requestRecovered)
                        recovered++;

                    var usable =
                        evaluations
                            .Where(x =>
                                x.BestText is not null &&
                                x.BestBubble is not null)
                            .ToList();

                    double textIoU =
                        usable.Count >= 2
                            ? IoU(
                                usable[0].BestText!.Bounds,
                                usable[1].BestText!.Bounds)
                            : 0;

                    double bubbleIoU =
                        usable.Count >= 2
                            ? IoU(
                                usable[0].BestBubble!.Bounds,
                                usable[1].BestBubble!.Bounds)
                            : 0;

                    bool twoScaleAgreement =
                        usable.Count >= 2 &&
                        AreSamePhysicalRegion(
                            usable[0].BestText!.Bounds,
                            usable[1].BestText!.Bounds,
                            0.45,
                            0.22) &&
                        AreSamePhysicalRegion(
                            usable[0].BestBubble!.Bounds,
                            usable[1].BestBubble!.Bounds,
                            0.30,
                            0.28);

                    float minTextScore =
                        usable.Count >= 2
                            ? usable
                                .Take(2)
                                .Min(x =>
                                    x.BestText!.Score)
                            : 0;

                    bool scoreStrong =
                        minTextScore >=
                        0.45f;

                    var chosen =
                        usable
                            .OrderByDescending(x =>
                                x.BestText!.Score)
                            .FirstOrDefault();

                    bool duplicateExisting =
                        chosen.BestText is not null &&
                        existingRegions
                            .Where(x =>
                                x.Kind ==
                                    PageRegionKind.TextBubble)
                            .Any(x =>
                                IsExistingTextDuplicate(
                                    chosen.BestText.Bounds,
                                    x.Bounds));

                    string? rejectReason =
                        !requestRecovered
                            ? "retry_not_recovered"
                            : usable.Count < 2
                                ? "missing_two_scale_parent"
                                : !twoScaleAgreement
                                    ? "cross_scale_inconsistent"
                                    : !scoreStrong
                                        ? "weak_retry_score"
                                        : duplicateExisting
                                            ? "duplicate_existing_fullpage"
                                            : null;

                    bool eligible =
                        rejectReason is null &&
                        chosen.BestText is not null &&
                        chosen.BestBubble is not null;

                    PageRegion? advisoryText =
                        null;

                    PageRegion? advisoryBubble =
                        null;

                    if (eligible)
                    {
                        advisoryEligible++;

                        var existingParent =
                            FindOwningBubble(
                                chosen.BestText!,
                                existingRegions);

                        advisoryBubble =
                            existingParent ??
                            chosen.BestBubble! with
                            {
                                RegionId =
                                    $"LRB{request.Region.Id:000}",
                                Source =
                                    "ogkalu-rtdetr-v2-laya-advisory"
                            };

                        advisoryText =
                            chosen.BestText! with
                            {
                                RegionId =
                                    $"LRT{request.Region.Id:000}",
                                Source =
                                    "ogkalu-rtdetr-v2-laya-advisory"
                            };

                        if (mode is
                                LayaDecisionMode.Advisory or
                                LayaDecisionMode.Active)
                        {
                            advisoryCandidates.Add(
                                new LayaDetectorAdvisoryCandidate(
                                    request.Region.Id,
                                    advisoryText,
                                    advisoryBubble,
                                    request.Confidence,
                                    textIoU,
                                    bubbleIoU,
                                    request.Region.Source.Text));
                        }
                    }

                    var passAudits =
                        evaluations
                            .Select(x =>
                                new
                                {
                                    x.Pass.Window.PassId,
                                    CropBounds =
                                        new
                                        {
                                            x.Pass.Window.CropBounds.X,
                                            x.Pass.Window.CropBounds.Y,
                                            x.Pass.Window.CropBounds.Width,
                                            x.Pass.Window.CropBounds.Height
                                        },
                                    x.Pass.Window.ContextMultiplier,
                                    x.Pass.Window.InputPixelsPerSourcePixel,
                                    RegionCount =
                                        x.Pass.Regions.Count,
                                    TextBubbleCount =
                                        x.Pass.Regions.Count(y =>
                                            y.Kind ==
                                                PageRegionKind.TextBubble),
                                    Recovered =
                                        x.BestText is not null,
                                    BestMatch =
                                        x.BestText is null
                                            ? null
                                            : new
                                            {
                                                x.BestText.RegionId,
                                                Kind =
                                                    x.BestText.Kind.ToString(),
                                                Bounds =
                                                    RectAudit(
                                                        x.BestText.Bounds),
                                                x.BestText.Score,
                                                x.BestText.Source,
                                                Affinity =
                                                    RtdetrPageRegionAnalyzer
                                                        .ShadowRetryAffinity(
                                                            request.Region.Source,
                                                            x.BestText)
                                            },
                                    ParentBubble =
                                        x.BestBubble is null
                                            ? null
                                            : new
                                            {
                                                x.BestBubble.RegionId,
                                                Bounds =
                                                    RectAudit(
                                                        x.BestBubble.Bounds),
                                                x.BestBubble.Score,
                                                x.BestBubble.Source
                                            }
                                })
                            .ToArray();

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
                            TwoScaleAgreement =
                                twoScaleAgreement,
                            CrossScaleTextIoU =
                                textIoU,
                            CrossScaleBubbleIoU =
                                bubbleIoU,
                            MinimumTextScore =
                                minTextScore,
                            ExistingFullPageDuplicate =
                                duplicateExisting,
                            AdvisoryEligible =
                                eligible,
                            Applied =
                                eligible &&
                                mode is
                                    LayaDecisionMode.Advisory or
                                    LayaDecisionMode.Active,
                            RejectReason =
                                rejectReason,
                            AdvisoryTextRegion =
                                advisoryText is null
                                    ? null
                                    : new
                                    {
                                        advisoryText.RegionId,
                                        Bounds =
                                            RectAudit(
                                                advisoryText.Bounds),
                                        advisoryText.Score,
                                        advisoryText.Source
                                    },
                            AdvisoryBubbleRegion =
                                advisoryBubble is null
                                    ? null
                                    : new
                                    {
                                        advisoryBubble.RegionId,
                                        Bounds =
                                            RectAudit(
                                                advisoryBubble.Bounds),
                                        advisoryBubble.Score,
                                        advisoryBubble.Source
                                    },
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
                            "pipeline-v2-laya-detector-retry-advisory-v1",
                        SourceFile =
                            Path.GetFileName(
                                sourcePath),
                        Mode =
                            mode.ToString(),
                        MutationApplied =
                            advisoryCandidates.Count >
                            0,
                        Requests =
                            requests.Count,
                        RecoveredRequests =
                            recovered,
                        AdvisoryEligibleRequests =
                            advisoryEligible,
                        AppliedRequests =
                            advisoryCandidates.Count,
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
                    $"Laya detector retry 완료 · recovered {recovered}/{requests.Count} · " +
                    $"advisory {advisoryCandidates.Count}/{advisoryEligible}");

                return new LayaDetectorRetryAuditResult(
                    advisoryCandidates);
            },
            token);
    }

    static object RectAudit(
        Rect rect)
        => new
        {
            rect.X,
            rect.Y,
            rect.Width,
            rect.Height
        };

    static PageRegion? FindOwningBubble(
        PageRegion text,
        IReadOnlyList<PageRegion> regions)
        => regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.Bubble)
            .Select(x =>
                new
                {
                    Region =
                        x,
                    Coverage =
                        Coverage(
                            text.Bounds,
                            x.Bounds),
                    CenterInside =
                        ContainsCenter(
                            text.Bounds,
                            x.Bounds)
                })
            .Where(x =>
                x.CenterInside ||
                x.Coverage >=
                    0.45)
            .OrderByDescending(x =>
                x.CenterInside)
            .ThenByDescending(x =>
                x.Coverage)
            .ThenByDescending(x =>
                x.Region.Score)
            .Select(x =>
                x.Region)
            .FirstOrDefault();

    public static bool AreSamePhysicalRegion(
        Rect a,
        Rect b,
        double minIoU,
        double maxNormalizedCenterDistance)
    {
        if (IoU(
                a,
                b) >=
            minIoU)
        {
            return true;
        }

        double ax =
            a.X +
            a.Width /
            2.0;

        double ay =
            a.Y +
            a.Height /
            2.0;

        double bx =
            b.X +
            b.Width /
            2.0;

        double by =
            b.Y +
            b.Height /
            2.0;

        double distance =
            Math.Sqrt(
                Math.Pow(
                    ax -
                    bx,
                    2) +
                Math.Pow(
                    ay -
                    by,
                    2));

        double scale =
            Math.Max(
                1.0,
                Math.Max(
                    Math.Max(
                        a.Width,
                        a.Height),
                    Math.Max(
                        b.Width,
                        b.Height)));

        return distance /
               scale <=
               maxNormalizedCenterDistance;
    }

    public static bool IsExistingTextDuplicate(
        Rect retry,
        Rect existing)
        => IoU(
               retry,
               existing) >=
               0.45 ||
           OverlapOverSmaller(
               retry,
               existing) >=
               0.75;

    static double Coverage(
        Rect inner,
        Rect outer)
    {
        double intersection =
            IntersectionArea(
                inner,
                outer);

        double area =
            Math.Max(
                1.0,
                inner.Width *
                (double)inner.Height);

        return intersection /
               area;
    }

    static bool ContainsCenter(
        Rect inner,
        Rect outer)
    {
        double cx =
            inner.X +
            inner.Width /
            2.0;

        double cy =
            inner.Y +
            inner.Height /
            2.0;

        return cx >=
                   outer.Left &&
               cx <=
                   outer.Right &&
               cy >=
                   outer.Top &&
               cy <=
                   outer.Bottom;
    }

    static double OverlapOverSmaller(
        Rect a,
        Rect b)
    {
        double intersection =
            IntersectionArea(
                a,
                b);

        double smaller =
            Math.Max(
                1.0,
                Math.Min(
                    a.Width *
                        (double)a.Height,
                    b.Width *
                        (double)b.Height));

        return intersection /
               smaller;
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
