using System.Text.Json;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

/// <summary>
/// Post-output diagnostics only. The final image is already saved before this
/// service runs, so audit failures never block delivery.
/// </summary>
public sealed class FinalAuditService
{
    const int PreviewMaxHeight = 1600;
    const int AuditWebpQuality = 82;

    sealed class AuditSummaryEntry
    {
        public string Page { get; set; } = "";
        public double ChangedRatio { get; set; }
        public long ChangedPixels { get; set; }
        public int RegionCount { get; set; }
        public int BubbleCount { get; set; }
        public int TextRegionCount { get; set; }
        public int CanonicalLineCount { get; set; }
        public int UnitCount { get; set; }
        public int SecondaryAccepted { get; set; }
        public int RenderApproved { get; set; }
        public int RenderRejected { get; set; }
    }

    public Task GenerateAsync(
        string sourcePath,
        string finalPath,
        string outputRoot,
        OcrStageResult stage,
        IReadOnlyList<VisionTranslation> translated,
        CancellationToken token = default)
        => Task.Run(
            () => Generate(
                sourcePath,
                finalPath,
                outputRoot,
                stage,
                translated,
                token),
            token);

    void Generate(
        string sourcePath,
        string finalPath,
        string outputRoot,
        OcrStageResult stage,
        IReadOnlyList<VisionTranslation> translated,
        CancellationToken token)
    {
        token.ThrowIfCancellationRequested();

        string auditDirectory =
            OutputDirectoryLayout.Audit(
                outputRoot);

        Directory.CreateDirectory(
            auditDirectory);

        string baseName =
            Path.GetFileNameWithoutExtension(
                sourcePath);

        using var original =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        using var final =
            Cv2.ImRead(
                finalPath,
                ImreadModes.Color);

        if (original.Empty() ||
            final.Empty())
        {
            throw new InvalidOperationException(
                "완료 검토를 위해 원본/결과 이미지를 열 수 없습니다.");
        }

        if (original.Size() !=
            final.Size())
        {
            throw new InvalidOperationException(
                "완료 검토 실패: 원본과 결과 이미지 크기가 다릅니다.");
        }

        using var difference =
            new Mat();

        Cv2.Absdiff(
            original,
            final,
            difference);

        using var gray =
            new Mat();

        Cv2.CvtColor(
            difference,
            gray,
            ColorConversionCodes.BGR2GRAY);

        using var changedMask =
            new Mat();

        Cv2.Threshold(
            gray,
            changedMask,
            8,
            255,
            ThresholdTypes.Binary);

        long changedPixels =
            Cv2.CountNonZero(
                changedMask);

        long totalPixels =
            Math.Max(
                1,
                original.Rows *
                (long)original.Cols);

        double changedRatio =
            changedPixels /
            (double)totalPixels;

        var changedRegions =
            FindChangedRegions(
                changedMask,
                token);

        string planPath =
            Path.Combine(
                OutputDirectoryLayout.Debug(
                    outputRoot),
                $"{baseName}.translated.render-plan.json");

        RenderPlanDocument? renderPlan =
            TryLoadPlan(
                planPath);

        SaveCompareImage(
            original,
            final,
            changedMask,
            Path.Combine(
                auditDirectory,
                $"{baseName}.final_compare.webp"));

        var regionAnalysis =
            stage.PageAnalysis;

        int regionCount =
            regionAnalysis?.Regions.Count ??
            0;

        int bubbleCount =
            regionAnalysis?.Regions.Count(x =>
                x.Kind ==
                PageRegionKind.Bubble) ??
            0;

        int textRegionCount =
            regionAnalysis?.Regions.Count(x =>
                x.Kind is
                    PageRegionKind.TextBubble or
                    PageRegionKind.TextFree) ??
            0;

        int renderApproved =
            renderPlan?.Units.Count(x =>
                x.Approved) ??
            0;

        int renderRejected =
            renderPlan?.Units.Count(x =>
                !x.Approved) ??
            0;

        var rejectionReasons =
            renderPlan?.Units
                .Where(x =>
                    !x.Approved)
                .GroupBy(x =>
                    x.Reason)
                .OrderByDescending(x =>
                    x.Count())
                .ToDictionary(
                    x => x.Key,
                    x => x.Count()) ??
            new Dictionary<string, int>();

        var passCounts =
            stage.Observations
                .GroupBy(x =>
                    x.Pass.ToString())
                .ToDictionary(
                    x => x.Key,
                    x => x.Count());

        var auditDocument =
            new
            {
                schema =
                    "final-audit-v1",
                source_file =
                    Path.GetFileName(
                        sourcePath),
                output_file =
                    Path.GetFileName(
                        finalPath),
                image =
                    new
                    {
                        width =
                            original.Cols,
                        height =
                            original.Rows,
                        changed_pixels =
                            changedPixels,
                        changed_ratio =
                            changedRatio,
                        changed_regions =
                            changedRegions
                    },
                region_analysis =
                    new
                    {
                        total =
                            regionCount,
                        bubbles =
                            bubbleCount,
                        text_regions =
                            textRegionCount,
                        mode =
                            regionAnalysis?.Mode,
                        external_detector =
                            regionAnalysis?.ExternalDetectorUsed
                    },
                ocr =
                    new
                    {
                        observations_by_pass =
                            passCounts,
                        canonical_lines =
                            stage.MergedLines.Count,
                        units =
                            stage.UnitBuild.Units.Count,
                        assigned_lines =
                            stage.UnitBuild.AssignedLineCount,
                        orphan_groups =
                            stage.UnitBuild.OrphanGroupCount,
                        secondary_total =
                            stage.SecondaryOcrEvidence.Count,
                        secondary_accepted =
                            stage.SecondaryOcrEvidence.Count(x =>
                                x.Accepted),
                        secondary_evidence =
                            stage.SecondaryOcrEvidence
                    },
                translation =
                    new
                    {
                        total =
                            translated.Count,
                        requested_render =
                            translated.Count(x =>
                                x.Render)
                    },
                render =
                    new
                    {
                        total =
                            renderPlan?.Units.Count ??
                            0,
                        approved =
                            renderApproved,
                        rejected =
                            renderRejected,
                        rejection_reasons =
                            rejectionReasons
                    }
            };

        string auditJsonPath =
            Path.Combine(
                auditDirectory,
                $"{baseName}.final_audit.json");

        File.WriteAllText(
            auditJsonPath,
            JsonSerializer.Serialize(
                auditDocument,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));

        UpdateSummary(
            auditDirectory,
            new AuditSummaryEntry
            {
                Page =
                    baseName,
                ChangedRatio =
                    changedRatio,
                ChangedPixels =
                    changedPixels,
                RegionCount =
                    regionCount,
                BubbleCount =
                    bubbleCount,
                TextRegionCount =
                    textRegionCount,
                CanonicalLineCount =
                    stage.MergedLines.Count,
                UnitCount =
                    stage.UnitBuild.Units.Count,
                SecondaryAccepted =
                    stage.SecondaryOcrEvidence.Count(x =>
                        x.Accepted),
                RenderApproved =
                    renderApproved,
                RenderRejected =
                    renderRejected
            });
    }

    static RenderPlanDocument? TryLoadPlan(
        string path)
    {
        try
        {
            if (!File.Exists(path))
                return null;

            return JsonSerializer.Deserialize<RenderPlanDocument>(
                File.ReadAllText(
                    path));
        }
        catch
        {
            return null;
        }
    }

    static IReadOnlyList<object> FindChangedRegions(
        Mat changedMask,
        CancellationToken token)
    {
        using var contourInput =
            changedMask.Clone();

        Cv2.FindContours(
            contourInput,
            out Point[][] contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        return contours
            .Select(x =>
            {
                token.ThrowIfCancellationRequested();
                return Cv2.BoundingRect(x);
            })
            .Where(x =>
                x.Width *
                (long)x.Height >=
                16)
            .OrderByDescending(x =>
                x.Width *
                (long)x.Height)
            .Take(30)
            .Select(x =>
                (object)new
                {
                    x = x.X,
                    y = x.Y,
                    width = x.Width,
                    height = x.Height
                })
            .ToList();
    }

    static void SaveCompareImage(
        Mat original,
        Mat final,
        Mat changedMask,
        string outputPath)
    {
        double scale =
            original.Rows > PreviewMaxHeight
                ? PreviewMaxHeight /
                  (double)original.Rows
                : 1.0;

        int width =
            Math.Max(
                1,
                (int)Math.Round(
                    original.Cols *
                    scale));

        int height =
            Math.Max(
                1,
                (int)Math.Round(
                    original.Rows *
                    scale));

        using var originalPreview =
            new Mat();

        using var finalPreview =
            new Mat();

        using var maskPreview =
            new Mat();

        Cv2.Resize(
            original,
            originalPreview,
            new Size(
                width,
                height),
            0,
            0,
            InterpolationFlags.Area);

        Cv2.Resize(
            final,
            finalPreview,
            new Size(
                width,
                height),
            0,
            0,
            InterpolationFlags.Area);

        Cv2.Resize(
            changedMask,
            maskPreview,
            new Size(
                width,
                height),
            0,
            0,
            InterpolationFlags.Nearest);

        using var diffPreview =
            originalPreview.Clone();

        using var red =
            new Mat(
                diffPreview.Size(),
                MatType.CV_8UC3,
                new Scalar(
                    0,
                    0,
                    255));

        using var overlay =
            new Mat();

        Cv2.AddWeighted(
            diffPreview,
            0.55,
            red,
            0.45,
            0,
            overlay);

        overlay.CopyTo(
            diffPreview,
            maskPreview);

        using var sheet =
            new Mat(
                height,
                width * 3,
                MatType.CV_8UC3,
                new Scalar(
                    20,
                    20,
                    20));

        originalPreview.CopyTo(
            new Mat(
                sheet,
                new Rect(
                    0,
                    0,
                    width,
                    height)));

        finalPreview.CopyTo(
            new Mat(
                sheet,
                new Rect(
                    width,
                    0,
                    width,
                    height)));

        diffPreview.CopyTo(
            new Mat(
                sheet,
                new Rect(
                    width * 2,
                    0,
                    width,
                    height)));

        DrawLabel(
            sheet,
            "ORIGINAL",
            12);

        DrawLabel(
            sheet,
            "FINAL",
            width + 12);

        DrawLabel(
            sheet,
            "DIFF",
            width * 2 + 12);

        Cv2.ImEncode(
            ".webp",
            sheet,
            out byte[] encoded,
            new[]
            {
                (int)ImwriteFlags.WebPQuality,
                AuditWebpQuality
            });

        File.WriteAllBytes(
            outputPath,
            encoded);
    }

    static void DrawLabel(
        Mat image,
        string text,
        int x)
    {
        Cv2.PutText(
            image,
            text,
            new Point(
                x,
                28),
            HersheyFonts.HersheySimplex,
            0.7,
            new Scalar(
                0,
                0,
                0),
            4,
            LineTypes.AntiAlias);

        Cv2.PutText(
            image,
            text,
            new Point(
                x,
                28),
            HersheyFonts.HersheySimplex,
            0.7,
            new Scalar(
                255,
                255,
                255),
            1,
            LineTypes.AntiAlias);
    }

    static void UpdateSummary(
        string auditDirectory,
        AuditSummaryEntry entry)
    {
        string path =
            Path.Combine(
                auditDirectory,
                "audit_summary.json");

        var entries =
            new List<AuditSummaryEntry>();

        try
        {
            if (File.Exists(path))
            {
                entries =
                    JsonSerializer.Deserialize<List<AuditSummaryEntry>>(
                        File.ReadAllText(path)) ??
                    [];
            }
        }
        catch
        {
            entries =
                [];
        }

        entries.RemoveAll(x =>
            string.Equals(
                x.Page,
                entry.Page,
                StringComparison.OrdinalIgnoreCase));

        entries.Add(
            entry);

        entries =
            entries
                .OrderByDescending(x =>
                    x.RenderRejected)
                .ThenByDescending(x =>
                    x.ChangedRatio)
                .ThenBy(x =>
                    x.Page,
                    StringComparer.OrdinalIgnoreCase)
                .ToList();

        File.WriteAllText(
            path,
            JsonSerializer.Serialize(
                entries,
                new JsonSerializerOptions
                {
                    WriteIndented = true
                }));
    }
}
