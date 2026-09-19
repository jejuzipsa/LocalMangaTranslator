using System.Text.Json;
using LocalMangaTranslator.Models;
using OpenCvSharp;

namespace LocalMangaTranslator.Services;

public static class PipelineDebugWriter
{
    public static void SavePreVisionDiagnostics(
        string sourcePath,
        string outputDirectory,
        OcrStageResult stage)
    {
        try
        {
            var debugDirectory =
                Path.Combine(
                    outputDirectory,
                    "debug");

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
                    $"{baseName}.00_page_candidates.png"));

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
                            x.Mask.Count(v => v != 0)
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
                    schema = "container-ocr-v1",
                    candidate_decisions = stage.CandidateDecisions,
                    attempts = stage.ContainerAttempts,
                    line_decisions = stage.ContainerLineDecisions
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
            debug);
    }
}

