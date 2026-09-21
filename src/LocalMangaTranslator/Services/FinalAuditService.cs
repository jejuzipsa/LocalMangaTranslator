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
        public int V2RequestedUnits { get; set; }
        public int V2TranslatedUnits { get; set; }
        public int V2PreservedOriginalUnits { get; set; }
        public int V2EraseCommittedUnits { get; set; }
        public int V2TypesetCommittedUnits { get; set; }
        public int V2MissingUnits { get; set; }
        public bool V2CountMatch { get; set; }
        public bool HasCommittedCleaned { get; set; }
        public long EraseChangedPixels { get; set; }
        public double EraseChangedRatio { get; set; }
        public long TypesetChangedPixels { get; set; }
        public double TypesetChangedRatio { get; set; }
        public int V2EraseReviewFailedUnits { get; set; }
        public int V2CleanedTextRedetectedUnits { get; set; }
        public int V2UnboundUnits { get; set; }
        public int V2DuplicateSuppressedUnits { get; set; }
        public int V2StylizedGraphicUnits { get; set; }
        public int V2OcrNoiseUnits { get; set; }
        public int V2OtherPreservedUnits { get; set; }
    }

    sealed class V2CommitUnitSummary
    {
        public int RegionId { get; set; }
        public string SourceText { get; set; } = "";
        public string Translation { get; set; } = "";
        public string Status { get; set; } = "";
        public bool Committed { get; set; }
        public bool Bound { get; set; }
        public bool LayoutApproved { get; set; }
        public bool EraseClean { get; set; }
        public bool LegacyEraseReviewClean { get; set; }
        public bool CleanedStateVerifierAvailable { get; set; }
        public bool CleanedStateEmpty { get; set; }
        public bool ReviewerDisagreement { get; set; }
        public string? LayoutMode { get; set; }
    }

    sealed class V2CommitAuditSummary
    {
        public int RequestedUnits { get; set; }
        public int TranslatedUnits { get; set; }
        public int PreservedOriginalUnits { get; set; }
        public int EraseCommittedUnits { get; set; }
        public int TypesetCommittedUnits { get; set; }
        public int MissingUnits { get; set; }
        public int DuplicateUnits { get; set; }
        public bool EraseTypesetCountMatch { get; set; }
        public bool SourceToFinalCountMatch { get; set; }
        public IReadOnlyList<V2CommitUnitSummary> Units { get; set; } =
            [];
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

        string debugDirectory =
            OutputDirectoryLayout.Debug(
                outputRoot);

        string planPath =
            Path.Combine(
                debugDirectory,
                $"{baseName}.translated.render-plan.json");

        RenderPlanDocument? renderPlan =
            TryLoadPlan(
                planPath);

        string v2CommitPath =
            Path.Combine(
                debugDirectory,
                $"{baseName}.v2_commit_audit.json");

        V2CommitAuditSummary? v2Commit =
            TryLoadV2CommitAudit(
                v2CommitPath);

        string committedCleanedPath =
            Path.Combine(
                debugDirectory,
                $"{baseName}.v2_05_committed_cleaned.webp");

        using var committedCleaned =
            File.Exists(
                committedCleanedPath)
                ? Cv2.ImRead(
                    committedCleanedPath,
                    ImreadModes.Color)
                : new Mat();

        bool hasCommittedCleaned =
            !committedCleaned.Empty() &&
            committedCleaned.Size() ==
                original.Size();

        using var eraseChangedMask =
            hasCommittedCleaned
                ? BuildChangedMask(
                    original,
                    committedCleaned)
                : Mat.Zeros(
                        original.Rows,
                        original.Cols,
                        MatType.CV_8UC1)
                    .ToMat();

        using var typesetChangedMask =
            hasCommittedCleaned
                ? BuildChangedMask(
                    committedCleaned,
                    final)
                : Mat.Zeros(
                        original.Rows,
                        original.Cols,
                        MatType.CV_8UC1)
                    .ToMat();

        long eraseChangedPixels =
            Cv2.CountNonZero(
                eraseChangedMask);

        long typesetChangedPixels =
            Cv2.CountNonZero(
                typesetChangedMask);

        double eraseChangedRatio =
            eraseChangedPixels /
            (double)totalPixels;

        double typesetChangedRatio =
            typesetChangedPixels /
            (double)totalPixels;

        SaveCompareImage(
            original,
            committedCleaned,
            hasCommittedCleaned,
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
                    "final-audit-v2",
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
                            changedRegions,
                        committed_cleaned_available =
                            hasCommittedCleaned,
                        committed_cleaned_file =
                            hasCommittedCleaned
                                ? Path.GetFileName(
                                    committedCleanedPath)
                                : null,
                        erase_changed_pixels =
                            eraseChangedPixels,
                        erase_changed_ratio =
                            eraseChangedRatio,
                        typeset_changed_pixels =
                            typesetChangedPixels,
                        typeset_changed_ratio =
                            typesetChangedRatio
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
                    },
                v2_commit =
                    v2Commit
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

        int v2EraseReviewFailed =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:erase_review_failed",
                    StringComparison.Ordinal)) ??
            0;

        int v2CleanedTextRedetected =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:cleaned_text_redetected",
                    StringComparison.Ordinal)) ??
            0;

        int v2Unbound =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:unbound",
                    StringComparison.Ordinal)) ??
            0;

        int v2DuplicateSuppressed =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:container_duplicate",
                    StringComparison.Ordinal)) ??
            0;

        int v2StylizedGraphic =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:stylized_graphic",
                    StringComparison.Ordinal)) ??
            0;

        int v2OcrNoise =
            v2Commit?.Units.Count(x =>
                string.Equals(
                    x.Status,
                    "original_preserved:ocr_noise",
                    StringComparison.Ordinal)) ??
            0;

        int v2OtherPreserved =
            Math.Max(
                0,
                (v2Commit?.PreservedOriginalUnits ?? 0) -
                v2EraseReviewFailed -
                v2CleanedTextRedetected -
                v2Unbound -
                v2DuplicateSuppressed -
                v2StylizedGraphic -
                v2OcrNoise);

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
                    renderRejected,
                V2RequestedUnits =
                    v2Commit?.RequestedUnits ?? 0,
                V2TranslatedUnits =
                    v2Commit?.TranslatedUnits ?? 0,
                V2PreservedOriginalUnits =
                    v2Commit?.PreservedOriginalUnits ?? 0,
                V2EraseCommittedUnits =
                    v2Commit?.EraseCommittedUnits ?? 0,
                V2TypesetCommittedUnits =
                    v2Commit?.TypesetCommittedUnits ?? 0,
                V2MissingUnits =
                    v2Commit?.MissingUnits ?? 0,
                V2CountMatch =
                    v2Commit is not null &&
                    v2Commit.EraseTypesetCountMatch &&
                    v2Commit.SourceToFinalCountMatch,
                HasCommittedCleaned =
                    hasCommittedCleaned,
                EraseChangedPixels =
                    eraseChangedPixels,
                EraseChangedRatio =
                    eraseChangedRatio,
                TypesetChangedPixels =
                    typesetChangedPixels,
                TypesetChangedRatio =
                    typesetChangedRatio,
                V2EraseReviewFailedUnits =
                    v2EraseReviewFailed,
                V2CleanedTextRedetectedUnits =
                    v2CleanedTextRedetected,
                V2UnboundUnits =
                    v2Unbound,
                V2DuplicateSuppressedUnits =
                    v2DuplicateSuppressed,
                V2StylizedGraphicUnits =
                    v2StylizedGraphic,
                V2OcrNoiseUnits =
                    v2OcrNoise,
                V2OtherPreservedUnits =
                    v2OtherPreserved
            });
    }

    static V2CommitAuditSummary? TryLoadV2CommitAudit(
        string path)
    {
        try
        {
            if (!File.Exists(path))
                return null;

            return JsonSerializer.Deserialize<V2CommitAuditSummary>(
                File.ReadAllText(
                    path));
        }
        catch
        {
            return null;
        }
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

    static Mat BuildChangedMask(
        Mat first,
        Mat second)
    {
        using var difference =
            new Mat();

        Cv2.Absdiff(
            first,
            second,
            difference);

        using var gray =
            new Mat();

        Cv2.CvtColor(
            difference,
            gray,
            ColorConversionCodes.BGR2GRAY);

        var mask =
            new Mat();

        Cv2.Threshold(
            gray,
            mask,
            8,
            255,
            ThresholdTypes.Binary);

        return mask;
    }

    static void SaveCompareImage(
        Mat original,
        Mat committedCleaned,
        bool hasCommittedCleaned,
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

        using var cleanedPreview =
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

        if (hasCommittedCleaned)
        {
            Cv2.Resize(
                committedCleaned,
                cleanedPreview,
                new Size(
                    width,
                    height),
                0,
                0,
                InterpolationFlags.Area);
        }

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

        int columns =
            hasCommittedCleaned
                ? 4
                : 3;

        using var sheet =
            new Mat(
                height,
                width *
                    columns,
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

        int finalColumn =
            hasCommittedCleaned
                ? 2
                : 1;

        int diffColumn =
            hasCommittedCleaned
                ? 3
                : 2;

        if (hasCommittedCleaned)
        {
            cleanedPreview.CopyTo(
                new Mat(
                    sheet,
                    new Rect(
                        width,
                        0,
                        width,
                        height)));
        }

        finalPreview.CopyTo(
            new Mat(
                sheet,
                new Rect(
                    width *
                        finalColumn,
                    0,
                    width,
                    height)));

        diffPreview.CopyTo(
            new Mat(
                sheet,
                new Rect(
                    width *
                        diffColumn,
                    0,
                    width,
                    height)));

        DrawLabel(
            sheet,
            "1 ORIGINAL",
            12);

        if (hasCommittedCleaned)
        {
            DrawLabel(
                sheet,
                "2 ERASE COMMIT",
                width + 12);

            DrawLabel(
                sheet,
                "3 FINAL",
                width * 2 + 12);

            DrawLabel(
                sheet,
                "4 DIFF",
                width * 3 + 12);
        }
        else
        {
            DrawLabel(
                sheet,
                "2 FINAL",
                width + 12);

            DrawLabel(
                sheet,
                "3 DIFF",
                width * 2 + 12);
        }

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
                    x.V2PreservedOriginalUnits)
                .ThenByDescending(x =>
                    x.V2EraseReviewFailedUnits)
                .ThenByDescending(x =>
                    x.V2UnboundUnits)
                .ThenByDescending(x =>
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
