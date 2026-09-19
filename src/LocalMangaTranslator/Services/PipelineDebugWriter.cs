using System.Text.Json;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public static class PipelineDebugWriter
{
    const int DebugWebpQuality = 82;

    public static void SavePageAnalysisDiagnostics(
        string sourcePath,
        string outputDirectory,
        PageAnalysisResult analysis)
    {
        try
        {
            string debugDirectory = Path.Combine(outputDirectory, "debug");
            Directory.CreateDirectory(debugDirectory);

            string baseName =
                Path.GetFileNameWithoutExtension(sourcePath) + ".translated";

            File.WriteAllText(
                Path.Combine(debugDirectory, $"{baseName}.00_region_analysis.json"),
                JsonSerializer.Serialize(new
                {
                    schema = "page-analysis-v3-region-first",
                    mode = analysis.Mode,
                    external_detector_used = analysis.ExternalDetectorUsed,
                    external_detector_status = analysis.ExternalDetectorStatus,
                    regions = analysis.Regions.Select(x => new
                    {
                        id = x.RegionId,
                        kind = x.Kind.ToString(),
                        x = x.Bounds.X,
                        y = x.Bounds.Y,
                        width = x.Bounds.Width,
                        height = x.Bounds.Height,
                        score = x.Score,
                        source = x.Source
                    }),
                    active_containers = analysis.ContainerCandidates.Select(x => new
                    {
                        id = x.CandidateId,
                        kind = x.Kind.ToString(),
                        x = x.Bounds.X,
                        y = x.Bounds.Y,
                        width = x.Bounds.Width,
                        height = x.Bounds.Height,
                        mode = x.DetectorMode,
                        score = x.Score,
                        fill_ratio = x.FillRatio
                    })
                }, new JsonSerializerOptions { WriteIndented = true }));

            using var source = Cv2.ImRead(sourcePath, ImreadModes.Color);
            if (source.Empty()) return;
            using var debug = source.Clone();

            foreach (var region in analysis.Regions)
            {
                Scalar color = region.Kind switch
                {
                    PageRegionKind.Bubble => new Scalar(0, 220, 0),
                    PageRegionKind.TextBubble => new Scalar(0, 0, 255),
                    PageRegionKind.TextFree => new Scalar(255, 0, 220),
                    PageRegionKind.Panel => new Scalar(255, 180, 0),
                    _ => new Scalar(180, 180, 180)
                };

                Cv2.Rectangle(debug, region.Bounds, color, 2);
                Cv2.PutText(
                    debug,
                    $"{region.RegionId}:{region.Kind}:{region.Score:0.00}",
                    new Point(region.Bounds.X, Math.Max(14, region.Bounds.Y - 3)),
                    HersheyFonts.HersheySimplex,
                    0.34,
                    color,
                    1,
                    LineTypes.AntiAlias);
            }

            Cv2.ImWrite(
                Path.Combine(debugDirectory, $"{baseName}.00_region_analysis.webp"),
                debug,
                new[]
                {
                    new ImageEncodingParam(
                        ImwriteFlags.WebPQuality,
                        DebugWebpQuality)
                });
        }
        catch
        {
            // Debug output must never fail page processing.
        }
    }
    public static void SavePreVisionDiagnostics(
        string sourcePath,
        string outputDirectory,
        OcrStageResult stage)
    {
        try
        {
            var debugDirectory =
                OutputDirectoryLayout.Debug(
                    outputDirectory);

            Directory.CreateDirectory(
                debugDirectory);

            string baseName =
                Path.GetFileNameWithoutExtension(
                    sourcePath) +
                ".translated";

            SavePageCandidateOverlay(
                sourcePath,
                stage.PageCandidates,
                Path.Combine(
                    debugDirectory,
                    $"{baseName}.00_page_candidates.webp"));

            var candidateDocument =
                stage.PageCandidates
                    .Select(x => new
                    {
                        id = x.CandidateId,
                        kind = x.Kind.ToString(),
                        x = x.Bounds.X,
                        y = x.Bounds.Y,
                        width = x.Bounds.Width,
                        height = x.Bounds.Height,
                        mode = x.DetectorMode,
                        dark = x.Dark,
                        score = x.Score,
                        fill_ratio = x.FillRatio,
                        border_touches = x.BorderTouches,
                        mask_pixels =
                            x.Mask.Count(v => v != 0),
                        region_id =
                            x.RegionId
                    })
                    .ToList();

            File.WriteAllText(
                Path.Combine(
                    debugDirectory,
                    $"{baseName}.00_page_candidates.json"),
                JsonSerializer.Serialize(
                    candidateDocument,
                    new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));

            File.WriteAllText(
                Path.Combine(debugDirectory, $"{baseName}.00_container_ocr_validation.json"),
                JsonSerializer.Serialize(new
                {
                    schema = "ocr-evidence-v3",
                    candidate_decisions = stage.CandidateDecisions,
                    attempts = stage.ContainerAttempts,
                    line_decisions = stage.ContainerLineDecisions,
                    region_attempts = stage.RegionAttempts,
                    region_line_decisions = stage.RegionLineDecisions,
                    secondary_ocr = stage.SecondaryOcrEvidence
                }, new JsonSerializerOptions { WriteIndented = true }));

            var observationDocument =
                stage.Observations
                    .Select(x => new
                    {
                        id = x.ObservationId,
                        pass = x.Pass.ToString(),
                        x = x.X,
                        y = x.Y,
                        width = x.W,
                        height = x.H,
                        text = x.Text,
                        confidence = x.Confidence,
                        language = x.Language,
                        scale = x.Scale,
                        source = x.SourceKey
                    })
                    .ToList();

            File.WriteAllText(
                Path.Combine(
                    debugDirectory,
                    $"{baseName}.00_ocr_observations.json"),
                JsonSerializer.Serialize(
                    observationDocument,
                    new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));

            SaveTrueOcrOverlay(
                sourcePath,
                stage.Observations
                    .Where(x =>
                        x.Pass is
                            OcrPassKind.Global1x or
                            OcrPassKind.Global2x or
                            OcrPassKind.Focus3x)
                    .ToList(),
                Path.Combine(
                    debugDirectory,
                    $"{baseName}.00_true_ocr.webp"));

            File.WriteAllText(
                Path.Combine(
                    debugDirectory,
                    $"{baseName}.00_line_ownership.json"),
                JsonSerializer.Serialize(
                    new
                    {
                        schema = "line-ownership-v1",
                        lines = stage.UnitBuild.LineOwnership,
                        units = stage.UnitBuild.UnitOwnership,
                        container_count = stage.UnitBuild.ContainerCount,
                        assigned_line_count = stage.UnitBuild.AssignedLineCount,
                        orphan_group_count = stage.UnitBuild.OrphanGroupCount
                    },
                    new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));
        }
        catch
        {
            // Debug output must never fail page processing.
        }
    }

    static void SaveTrueOcrOverlay(
        string sourcePath,
        IReadOnlyList<OcrObservation> observations,
        string outputPath)
    {
        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            return;

        using var debug =
            source.Clone();

        foreach (var observation in observations)
        {
            var rect =
                new Rect(
                    Math.Max(
                        0,
                        (int)Math.Floor(
                            observation.X)),
                    Math.Max(
                        0,
                        (int)Math.Floor(
                            observation.Y)),
                    Math.Max(
                        1,
                        (int)Math.Ceiling(
                            observation.W)),
                    Math.Max(
                        1,
                        (int)Math.Ceiling(
                            observation.H)));

            Scalar color =
                observation.Pass switch
                {
                    OcrPassKind.Global1x =>
                        new Scalar(
                            0,
                            220,
                            0),
                    OcrPassKind.Global2x =>
                        new Scalar(
                            0,
                            180,
                            255),
                    OcrPassKind.Focus3x =>
                        new Scalar(
                            255,
                            0,
                            220),
                    _ =>
                        new Scalar(
                            180,
                            180,
                            180)
                };

            Cv2.Rectangle(
                debug,
                rect,
                color,
                1);

            string label =
                $"{observation.Pass}:{observation.Confidence:0.00}:{CompactLabel(observation.Text)}";

            Cv2.PutText(
                debug,
                label,
                new Point(
                    rect.X,
                    Math.Max(
                        12,
                        rect.Y - 2)),
                HersheyFonts.HersheySimplex,
                0.30,
                color,
                1,
                LineTypes.AntiAlias);
        }

        Cv2.ImWrite(
            outputPath,
            debug,
            new[]
            {
                new ImageEncodingParam(
                    ImwriteFlags.WebPQuality,
                    DebugWebpQuality)
            });
    }

    static string CompactLabel(
        string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "";

        string oneLine =
            value
                .Replace(
                    '\r',
                    ' ')
                .Replace(
                    '\n',
                    ' ')
                .Trim();

        return oneLine.Length <= 26
            ? oneLine
            : oneLine[..26] + "…";
    }

    static void SavePageCandidateOverlay(
        string sourcePath,
        IReadOnlyList<ContainerCandidate> candidates,
        string outputPath)
    {
        using var source =
            Cv2.ImRead(
                sourcePath,
                ImreadModes.Color);

        if (source.Empty())
            return;

        using var debug =
            source.Clone();

        foreach (var candidate in candidates)
        {
            Scalar color =
                candidate.Kind switch
                {
                    ContainerCandidateKind.Caption =>
                        new Scalar(
                            0,
                            180,
                            255),

                    ContainerCandidateKind.Speech =>
                        candidate.Dark
                            ? new Scalar(
                                255,
                                170,
                                0)
                            : new Scalar(
                                0,
                                210,
                                0),

                    _ =>
                        new Scalar(
                            180,
                            180,
                            180)
                };

            Cv2.Rectangle(
                debug,
                candidate.Bounds,
                color,
                2);

            Cv2.PutText(
                debug,
                $"{candidate.CandidateId}:{candidate.DetectorMode}",
                new Point(
                    candidate.Bounds.X,
                    Math.Max(
                        14,
                        candidate.Bounds.Y - 3)),
                HersheyFonts.HersheySimplex,
                0.34,
                color,
                1,
                LineTypes.AntiAlias);
        }

        Cv2.ImWrite(
            outputPath,
            debug,
            new[]
            {
                new ImageEncodingParam(
                    ImwriteFlags.WebPQuality,
                    DebugWebpQuality)
            });
    }
}

