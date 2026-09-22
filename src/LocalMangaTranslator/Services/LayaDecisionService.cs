using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.PipelineV2.Erase;

namespace LocalMangaTranslator.Services;

public sealed record LayaPythonRuntime(
    string FileName,
    string[] PrefixArguments,
    string DisplayName);

public sealed class LayaPythonNotFoundException : InvalidOperationException
{
    public LayaPythonNotFoundException(string message)
        : base(message)
    {
    }
}

public sealed record LayaDetectorRetryRequest(
    VisionTranslation Region,
    double? Confidence);

public sealed record LayaRenderRecoveryRequest(
    VisionTranslation Region,
    double? Confidence);

public sealed record LayaShadowAuditResult(
    IReadOnlyList<LayaDetectorRetryRequest> DetectorRetryRequests,
    IReadOnlyList<LayaRenderRecoveryRequest> RenderRecoveryRequests)
{
    public static LayaShadowAuditResult Empty { get; } =
        new([], []);
}

/// <summary>
/// 0036/0037 experimental System-1 decision track.
///
/// Laya never creates geometry and never mutates the Classic result in Shadow
/// mode. It receives compact evidence only, then records typed decisions beside
/// the normal debug artifacts. A persistent Python worker keeps the checkpoint
/// resident so one page can ask many small questions without reloading it.
/// </summary>
public sealed class LayaDecisionService
{
    const string LayaPackageVersion = "0.3.5";

    readonly SemaphoreSlim startGate = new(1, 1);
    readonly SemaphoreSlim ioGate = new(1, 1);

    Process? worker;
    StreamWriter? workerInput;
    StreamReader? workerOutput;
    Task<string>? workerErrorDrain;
    bool packageChecked;
    string? runtimeDescription;
    LayaPythonRuntime? pythonRuntime;

    public bool IsReady =>
        worker is not null &&
        !worker.HasExited &&
        workerInput is not null &&
        workerOutput is not null;

    public string? RuntimeDescription =>
        runtimeDescription;

    public async Task<string> PrepareAsync(
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        await EnsureWorkerAsync(
            progress,
            token);

        return runtimeDescription ??
               "Laya ready";
    }

    public static IReadOnlyList<LayaPythonRuntime> BuildPythonCandidates(
        string? configured = null,
        string? localAppData = null,
        string? programFiles = null)
    {
        configured ??=
            Environment.GetEnvironmentVariable(
                "LMT_LAYA_PYTHON");

        localAppData ??=
            Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData);

        programFiles ??=
            Environment.GetFolderPath(
                Environment.SpecialFolder.ProgramFiles);

        var result =
            new List<LayaPythonRuntime>();

        var seen =
            new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);

        void Add(
            string? fileName,
            string[] prefixArguments,
            string displayName)
        {
            if (string.IsNullOrWhiteSpace(
                    fileName))
            {
                return;
            }

            string normalized =
                fileName
                    .Trim()
                    .Trim('"');

            string key =
                normalized +
                "\u001f" +
                string.Join(
                    "\u001f",
                    prefixArguments);

            if (!seen.Add(
                    key))
            {
                return;
            }

            result.Add(
                new LayaPythonRuntime(
                    normalized,
                    prefixArguments,
                    displayName));
        }

        Add(
            configured,
            Array.Empty<string>(),
            "LMT_LAYA_PYTHON");

        Add(
            "python",
            Array.Empty<string>(),
            "python");

        Add(
            "py",
            new[] { "-3" },
            "py -3");

        Add(
            "python3",
            Array.Empty<string>(),
            "python3");

        foreach (string version in
                 new[]
                 {
                     "313",
                     "312",
                     "311",
                     "310"
                 })
        {
            if (!string.IsNullOrWhiteSpace(
                    localAppData))
            {
                Add(
                    Path.Combine(
                        localAppData,
                        "Programs",
                        "Python",
                        $"Python{version}",
                        "python.exe"),
                    Array.Empty<string>(),
                    $"Python {version[..1]}.{version[1..]} (LocalAppData)");
            }

            if (!string.IsNullOrWhiteSpace(
                    programFiles))
            {
                Add(
                    Path.Combine(
                        programFiles,
                        $"Python{version}",
                        "python.exe"),
                    Array.Empty<string>(),
                    $"Python {version[..1]}.{version[1..]} (Program Files)");
            }
        }

        return result;
    }

    public static void ConfigurePythonEnvironment(
        ProcessStartInfo start)
    {
        start.Environment["USE_TF"] =
            "0";

        start.Environment["HF_HUB_DISABLE_SYMLINKS"] =
            "1";

        start.Environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] =
            "1";

        start.Environment["PYTHONUTF8"] =
            "1";

        start.Environment["PYTHONIOENCODING"] =
            "utf-8";
    }

    public static IReadOnlyList<IReadOnlyList<int>> FindV2DuplicateCandidateGroups(
        IReadOnlyList<VisionTranslation> translated,
        V2EraseSelection selection)
        => BuildV2DuplicateGroups(
                translated,
                selection)
            .Select(x =>
                (IReadOnlyList<int>)x.Candidates
                    .Select(y => y.Id)
                    .OrderBy(y => y)
                    .ToArray())
            .ToArray();

    static List<LayaV2DuplicateGroup> BuildV2DuplicateGroups(
        IReadOnlyList<VisionTranslation> translated,
        V2EraseSelection? selection)
    {
        if (selection is null)
            return [];

        return translated
            .Where(x =>
                x.Render &&
                !string.IsNullOrWhiteSpace(
                    x.Translation) &&
                selection.Bindings.ContainsKey(
                    x.Id))
            .Select(x =>
            {
                var binding =
                    selection.Bindings[x.Id];

                return new
                {
                    Region =
                        x,
                    Binding =
                        binding,
                    Key =
                        V2PhysicalOwnerKey(
                            binding)
                };
            })
            .GroupBy(
                x =>
                    x.Key,
                StringComparer.Ordinal)
            .Where(x =>
                x.Count() > 1)
            .Select(x =>
                new LayaV2DuplicateGroup(
                    x.Key,
                    x.Select(y =>
                            y.Region)
                        .OrderBy(y =>
                            y.Id)
                        .ToList()))
            .OrderBy(x =>
                x.PhysicalOwnerKey,
                StringComparer.Ordinal)
            .ToList();
    }

    static string V2PhysicalOwnerKey(
        V2RegionBinding binding)
    {
        if (!string.IsNullOrWhiteSpace(
                binding.BubbleRegionId))
        {
            return "bubble:" +
                   binding.BubbleRegionId;
        }

        var r =
            binding.LayoutBounds;

        return
            $"layout:{r.X},{r.Y},{r.Width},{r.Height}";
    }

    static string NormalizeOcrAgreement(
        string? agreement)
    {
        string value =
            agreement?
                .Trim()
                .ToLowerInvariant() ??
            "";

        return value switch
        {
            "agree" =>
                "agree",
            "partial" =>
                "partial",
            "disagree" =>
                "disagree",
            "secondary_only" =>
                "secondary_only",
            _ =>
                "unknown"
        };
    }

    static string NormalizeComparableText(
        string? text)
        => new(
            (text ?? "")
                .Where(x =>
                    !char.IsWhiteSpace(
                        x))
                .Select(char.ToLowerInvariant)
                .ToArray());

    public async Task<LayaShadowAuditResult> WriteShadowAuditAsync(
        string sourcePath,
        string outputDirectory,
        PageAnalysisResult analysis,
        OcrStageResult ocrStage,
        IReadOnlyList<VisionTranslation> visionReviewedBeforeRecovery,
        IReadOnlyList<VisionTranslation> reviewedAfterRecovery,
        IReadOnlyList<VisionTranslation> translated,
        V2EraseSelection? v2Selection,
        PipelineOptions options,
        IProgress<string>? progress = null,
        CancellationToken token = default)
    {
        if (options.DecisionTrack !=
                PipelineDecisionTrack.LayaExperimental ||
            options.LayaMode ==
                LayaDecisionMode.Off)
        {
            return LayaShadowAuditResult.Empty;
        }

        string page =
            Path.GetFileNameWithoutExtension(
                sourcePath);

        string decisionPath =
            Path.Combine(
                OutputDirectoryLayout.Debug(
                    outputDirectory),
                page +
                ".laya_decisions.json");

        string summaryPath =
            Path.Combine(
                OutputDirectoryLayout.Debug(
                    outputDirectory),
                page +
                ".laya_summary.json");

        var decisions =
            new List<LayaShadowDecision>();

        var detectorRetryRequests =
            new List<LayaDetectorRetryRequest>();

        var renderRecoveryRequests =
            new List<LayaRenderRecoveryRequest>();

        string? runtimeError =
            null;

        try
        {
            progress?.Report(
                "LayaExperimental · decision worker 준비");

            await EnsureWorkerAsync(
                progress,
                token);

            foreach (var region in translated)
            {
                token.ThrowIfCancellationRequested();

                var candidates =
                    BuildOwnershipCandidates(
                        region.Source,
                        analysis.Regions);

                string ruleOwner =
                    region.Source.RegionTextRegion?.RegionId ??
                    "UNBOUND";

                if (candidates.Count > 0)
                {
                    var criteria =
                        candidates.ToDictionary(
                            x => x.TextRegionId,
                            x =>
                                $"RT-DETR TextBubble; parent={x.BubbleRegionId ?? "none"}; " +
                                $"detector_score={x.DetectorScore:0.000}; " +
                                $"block_coverage={x.BlockCoverage:0.000}; " +
                                $"center_inside={x.CenterInside}; " +
                                $"center_distance={x.CenterDistance:0.0}",
                            StringComparer.Ordinal);

                    criteria["UNBOUND"] =
                        "No detector TextBubble has sufficient physical evidence for this OCR unit.";

                    var state =
                        new
                        {
                            task =
                                "comic_ocr_physical_owner",
                            source_text =
                                region.Source.Text,
                            corrected_text =
                                region.CorrectedText,
                            current_rule_owner =
                                ruleOwner,
                            current_parent_bubble =
                                region.Source.RegionId,
                            ocr_confidence =
                                AverageConfidence(
                                    region.Source),
                            secondary_ocr =
                                region.Source.SecondaryOcrText,
                            secondary_agreement =
                                region.Source.SecondaryOcrAgreement,
                            block =
                                new
                                {
                                    region.Source.X,
                                    region.Source.Y,
                                    region.Source.W,
                                    region.Source.H
                                },
                            detector_candidates =
                                candidates
                        };

                    var questions =
                        new Dictionary<string, object>(
                            StringComparer.Ordinal)
                        {
                            ["owner"] =
                                new
                                {
                                    type = "choice",
                                    instructions =
                                        "Choose the single RT-DETR TextBubble that physically owns this OCR text. " +
                                        "Prefer direct geometry, center containment, OCR agreement and detector evidence. " +
                                        "Choose UNBOUND when detector geometry does not support a safe owner. " +
                                        "Do not invent geometry.",
                                    criteria
                                }
                        };

                    var response =
                        await PredictAsync(
                            state,
                            questions,
                            token);

                    var answer =
                        ReadChoice(
                            response,
                            "owner");

                    decisions.Add(
                        new LayaShadowDecision
                        {
                            Kind =
                                "ownership",
                            RegionId =
                                region.Id,
                            RuleDecision =
                                ruleOwner,
                            LayaDecision =
                                answer.Value,
                            Confidence =
                                answer.Confidence,
                            Match =
                                answer.Value is null
                                    ? null
                                    : string.Equals(
                                        ruleOwner,
                                        answer.Value,
                                        StringComparison.Ordinal),
                            Status =
                                answer.Value is null
                                    ? "laya_no_answer"
                                    : "shadow_only",
                            Evidence =
                                new
                                {
                                    candidates,
                                    region.Source.RegionId,
                                    region.Source.SecondaryOcrAgreement
                                }
                        });
                }
                else if (ruleOwner == "UNBOUND")
                {
                    var state =
                        new
                        {
                            task =
                                "comic_detector_retry",
                            source_text =
                                region.Source.Text,
                            corrected_text =
                                region.CorrectedText,
                            ocr_confidence =
                                AverageConfidence(
                                    region.Source),
                            secondary_ocr =
                                region.Source.SecondaryOcrText,
                            secondary_agreement =
                                region.Source.SecondaryOcrAgreement,
                            block =
                                new
                                {
                                    region.Source.X,
                                    region.Source.Y,
                                    region.Source.W,
                                    region.Source.H
                                },
                            detector_textbubble_candidates =
                                0
                        };

                    var response =
                        await PredictAsync(
                            state,
                            new Dictionary<string, object>
                            {
                                ["retry_detector"] =
                                    new
                                    {
                                        type = "noul",
                                        instructions =
                                            "Should the structural text detector be retried at another image scale? " +
                                            "Return true only when OCR evidence strongly supports real readable comic text " +
                                            "but this unit has no RT-DETR TextBubble owner. " +
                                            "This decision may request another detector pass but must not create geometry."
                                    }
                            },
                            token);

                    var answer =
                        ReadBoolean(
                            response,
                            "retry_detector");

                    decisions.Add(
                        new LayaShadowDecision
                        {
                            Kind =
                                "detector_retry",
                            RegionId =
                                region.Id,
                            RuleDecision =
                                "false",
                            LayaDecision =
                                answer.Value.HasValue
                                    ? answer.Value.Value
                                        ? "true"
                                        : "false"
                                    : null,
                            Confidence =
                                answer.Confidence,
                            Match =
                                answer.Value.HasValue
                                    ? !answer.Value.Value
                                    : null,
                            Status =
                                answer.Value.HasValue
                                    ? "shadow_only"
                                    : "laya_no_answer",
                            Evidence =
                                state
                        });

                    if (answer.Value == true)
                    {
                        detectorRetryRequests.Add(
                            new LayaDetectorRetryRequest(
                                region,
                                answer.Confidence));
                    }
                }
            }

            foreach (var region in
                     visionReviewedBeforeRecovery.Where(x =>
                         !x.Render &&
                         x.Type is
                             "dialogue" or
                             "thought" or
                             "caption"))
            {
                token.ThrowIfCancellationRequested();

                bool ruleRecover =
                    PagePipelineService
                        .ShouldRecoverDetectorOwnedRenderableUnit(
                            region,
                            analysis.Regions);

                var state =
                    new
                    {
                        task =
                            "comic_render_recovery",
                        stage =
                            "vision_pre_recovery",
                        source_text =
                            region.Source.Text,
                        corrected_text =
                            region.CorrectedText,
                        type =
                            region.Type,
                        vision_render =
                            region.Render,
                        ocr_confidence =
                            AverageConfidence(
                                region.Source),
                        secondary_ocr =
                            region.Source.SecondaryOcrText,
                        secondary_agreement =
                            region.Source.SecondaryOcrAgreement,
                        bubble_region_id =
                            region.Source.RegionId,
                        text_region_id =
                            region.Source.RegionTextRegion?.RegionId,
                        detector_text_candidates =
                            BuildOwnershipCandidates(
                                region.Source,
                                analysis.Regions)
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["recover_render"] =
                                new
                                {
                                    type = "noul",
                                    instructions =
                                        "Should this detector-supported dialogue/thought/caption be re-armed for final translation " +
                                        "after Vision returned render=false? " +
                                        "Use OCR strength, independent OCR agreement and immutable detector evidence. " +
                                        "Return false for noise, logos, signs, SFX or unsupported text."
                                }
                        },
                        token);

                var answer =
                    ReadBoolean(
                        response,
                        "recover_render");

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "render_recovery",
                        RegionId =
                            region.Id,
                        RuleDecision =
                            ruleRecover
                                ? "true"
                                : "false",
                        LayaDecision =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                    ? "true"
                                    : "false"
                                : null,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value.HasValue
                                ? ruleRecover ==
                                  answer.Value.Value
                                : null,
                        Status =
                            answer.Value.HasValue
                                ? "shadow_only"
                                : "laya_no_answer",
                        Evidence =
                            state
                    });
            }

            foreach (var region in
                     translated.Where(x =>
                         !x.Render &&
                         x.Type is
                             "dialogue" or
                             "thought" or
                             "caption"))
            {
                token.ThrowIfCancellationRequested();

                var enteredTranslation =
                    reviewedAfterRecovery
                        .FirstOrDefault(x =>
                            x.Id ==
                            region.Id);

                bool classicRecoveryEligible =
                    PagePipelineService
                        .ShouldRecoverDetectorOwnedRenderableUnit(
                            region,
                            analysis.Regions);

                var state =
                    new
                    {
                        task =
                            "comic_render_recovery",
                        stage =
                            "post_translation_final",
                        source_text =
                            region.Source.Text,
                        corrected_text =
                            region.CorrectedText,
                        type =
                            region.Type,
                        entered_translation_renderable =
                            enteredTranslation?.Render,
                        final_render =
                            region.Render,
                        classic_recovery_eligible =
                            classicRecoveryEligible,
                        ocr_confidence =
                            AverageConfidence(
                                region.Source),
                        secondary_ocr =
                            region.Source.SecondaryOcrText,
                        secondary_agreement =
                            region.Source.SecondaryOcrAgreement,
                        bubble_region_id =
                            region.Source.RegionId,
                        text_region_id =
                            region.Source.RegionTextRegion?.RegionId,
                        detector_text_candidates =
                            BuildOwnershipCandidates(
                                region.Source,
                                analysis.Regions)
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["recover_render"] =
                                new
                                {
                                    type = "noul",
                                    instructions =
                                        "This dialogue/thought/caption reached the final post-translation state as render=false. " +
                                        "Should it be re-armed instead? Judge only whether the text is a real translatable comic unit " +
                                        "supported by OCR and immutable detector evidence. Return false for noise, SFX, logos or unsupported text."
                                }
                        },
                        token);

                var answer =
                    ReadBoolean(
                        response,
                        "recover_render");

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "render_recovery",
                        RegionId =
                            region.Id,
                        RuleDecision =
                            "false",
                        LayaDecision =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                    ? "true"
                                    : "false"
                                : null,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value.HasValue
                                ? !answer.Value.Value
                                : null,
                        Status =
                            answer.Value.HasValue
                                ? "shadow_only"
                                : "laya_no_answer",
                        Evidence =
                            state
                    });

                if (answer.Value == true)
                {
                    renderRecoveryRequests.Add(
                        new LayaRenderRecoveryRequest(
                            region,
                            answer.Confidence));
                }
            }

            var duplicateGroups =
                BuildV2DuplicateGroups(
                    translated,
                    v2Selection);

            foreach (var group in
                     duplicateGroups)
            {
                token.ThrowIfCancellationRequested();

                var candidates =
                    group.Candidates
                        .OrderBy(x =>
                            x.Id)
                        .ToList();

                var classic =
                    candidates
                        .OrderByDescending(x =>
                            RenderPipelineService
                                .GetV2CandidateKeepScore(
                                    x))
                        .ThenBy(x =>
                            x.Id)
                        .First();

                var criteria =
                    candidates.ToDictionary(
                        x =>
                            $"R{x.Id}",
                        x =>
                            $"source={Compact(x.Source.Text)}; corrected={Compact(x.CorrectedText)}; " +
                            $"secondary_agreement={x.Source.SecondaryOcrAgreement ?? "none"}; " +
                            $"ocr_confidence={AverageConfidence(x.Source):0.000}; " +
                            $"detector_text_region={x.Source.RegionTextRegion?.RegionId ?? "none"}; " +
                            $"classic_keep_score={RenderPipelineService.GetV2CandidateKeepScore(x):0.0}",
                        StringComparer.Ordinal);

                var state =
                    new
                    {
                        task =
                            "comic_duplicate_arbitration",
                        stage =
                            "v2_physical_container",
                        physical_owner_key =
                            group.PhysicalOwnerKey,
                        candidates =
                            candidates.Select(x =>
                            {
                                var binding =
                                    v2Selection!.Bindings[
                                        x.Id];

                                return new
                                {
                                    option =
                                        $"R{x.Id}",
                                    x.Id,
                                    source =
                                        x.Source.Text,
                                    corrected =
                                        x.CorrectedText,
                                    secondary_agreement =
                                        x.Source.SecondaryOcrAgreement,
                                    ocr_confidence =
                                        AverageConfidence(
                                            x.Source),
                                    detector_text_region =
                                        x.Source.RegionTextRegion?.RegionId,
                                    binding.TextRegionIds,
                                    binding.BubbleRegionId,
                                    binding.LayoutBounds,
                                    classic_keep_score =
                                        RenderPipelineService
                                            .GetV2CandidateKeepScore(
                                                x)
                                };
                            })
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["duplicate_winner"] =
                                new
                                {
                                    type = "choice",
                                    instructions =
                                        "These candidates map to the same immutable V2 physical comic container. " +
                                        "Choose the single candidate that should survive duplicate arbitration. " +
                                        "Prefer faithful complete OCR, independent OCR agreement, exact detector ownership and no hallucinated expansion.",
                                    criteria
                                }
                        },
                        token);

                var answer =
                    ReadChoice(
                        response,
                        "duplicate_winner");

                string classicDecision =
                    $"R{classic.Id}";

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "duplicate",
                        RegionId =
                            classic.Id,
                        RuleDecision =
                            classicDecision,
                        LayaDecision =
                            answer.Value,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value is null
                                ? null
                                : string.Equals(
                                    classicDecision,
                                    answer.Value,
                                    StringComparison.Ordinal),
                        Status =
                            answer.Value is null
                                ? "laya_no_answer"
                                : "shadow_only",
                        Evidence =
                            state
                    });
            }

            foreach (var region in
                     translated.Where(x =>
                         !string.IsNullOrWhiteSpace(
                             x.Source.SecondaryOcrText)))
            {
                token.ThrowIfCancellationRequested();

                string ruleAgreement =
                    NormalizeOcrAgreement(
                        region.Source.SecondaryOcrAgreement);

                var state =
                    new
                    {
                        task =
                            "comic_ocr_consensus",
                        primary_ocr =
                            region.Source.Text,
                        secondary_ocr =
                            region.Source.SecondaryOcrText,
                        primary_confidence =
                            AverageConfidence(
                                region.Source),
                        current_agreement =
                            ruleAgreement,
                        corrected_text =
                            region.CorrectedText
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["ocr_consensus"] =
                                new
                                {
                                    type = "choice",
                                    instructions =
                                        "Classify the relation between primary and independent secondary OCR. " +
                                        "Choose agree when they express the same full reading, partial when there is meaningful overlap " +
                                        "with omissions/additions, disagree when the readings materially conflict, secondary_only when " +
                                        "the primary has no meaningful reading, otherwise unknown.",
                                    criteria =
                                        new Dictionary<string, string>(
                                            StringComparer.Ordinal)
                                        {
                                            ["agree"] =
                                                "same full text after harmless punctuation/spacing differences",
                                            ["partial"] =
                                                "substantial shared reading but one side is incomplete or has meaningful extras",
                                            ["disagree"] =
                                                "materially conflicting reading",
                                            ["secondary_only"] =
                                                "secondary OCR contains the meaningful reading while primary is effectively absent/noise",
                                            ["unknown"] =
                                                "insufficient evidence for the other labels"
                                        }
                                }
                        },
                        token);

                var answer =
                    ReadChoice(
                        response,
                        "ocr_consensus");

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "ocr_consensus",
                        RegionId =
                            region.Id,
                        RuleDecision =
                            ruleAgreement,
                        LayaDecision =
                            answer.Value,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value is null
                                ? null
                                : string.Equals(
                                    ruleAgreement,
                                    answer.Value,
                                    StringComparison.Ordinal),
                        Status =
                            answer.Value is null
                                ? "laya_no_answer"
                                : "shadow_only",
                        Evidence =
                            state
                    });
            }

            foreach (var region in
                     translated.Where(x =>
                         NormalizeComparableText(
                             x.Source.Text) !=
                         NormalizeComparableText(
                             x.CorrectedText)))
            {
                token.ThrowIfCancellationRequested();

                var state =
                    new
                    {
                        task =
                            "comic_vision_correction",
                        source_text =
                            region.Source.Text,
                        corrected_text =
                            region.CorrectedText,
                        secondary_ocr =
                            region.Source.SecondaryOcrText,
                        secondary_agreement =
                            region.Source.SecondaryOcrAgreement,
                        ocr_confidence =
                            AverageConfidence(
                                region.Source),
                        type =
                            region.Type
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["accept_correction"] =
                                new
                                {
                                    type = "noul",
                                    instructions =
                                        "Should the Vision-corrected comic text be accepted as a faithful correction of OCR? " +
                                        "Use the primary OCR, independent secondary OCR and their confidence/agreement. " +
                                        "Return false for unsupported invented words, large hallucinated expansions or corrections " +
                                        "that contradict stronger OCR evidence."
                                }
                        },
                        token);

                var answer =
                    ReadBoolean(
                        response,
                        "accept_correction");

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "vision_correction",
                        RegionId =
                            region.Id,
                        RuleDecision =
                            "accept",
                        LayaDecision =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                    ? "accept"
                                    : "reject"
                                : null,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                : null,
                        Status =
                            answer.Value.HasValue
                                ? "shadow_only"
                                : "laya_no_answer",
                        Evidence =
                            state
                    });
            }

            foreach (var region in
                     translated.Where(x =>
                         x.Render &&
                         !string.IsNullOrWhiteSpace(
                             x.Translation) &&
                         x.Type is
                             "dialogue" or
                             "thought" or
                             "caption"))
            {
                token.ThrowIfCancellationRequested();

                var state =
                    new
                    {
                        task =
                            "comic_translation_consistency",
                        source_text =
                            region.Source.Text,
                        corrected_source_text =
                            region.CorrectedText,
                        korean_translation =
                            region.Translation,
                        type =
                            region.Type,
                        secondary_ocr =
                            region.Source.SecondaryOcrText,
                        secondary_agreement =
                            region.Source.SecondaryOcrAgreement
                    };

                var response =
                    await PredictAsync(
                        state,
                        new Dictionary<string, object>
                        {
                            ["translation_consistent"] =
                                new
                                {
                                    type = "noul",
                                    instructions =
                                        "Does the Korean translation preserve the corrected source meaning closely enough for comic rendering? " +
                                        "Check major omissions/additions, negation, names, intent and sentence polarity. " +
                                        "Judge semantic faithfulness, not stylistic preference. Return false for a materially wrong translation."
                                }
                        },
                        token);

                var answer =
                    ReadBoolean(
                        response,
                        "translation_consistent");

                decisions.Add(
                    new LayaShadowDecision
                    {
                        Kind =
                            "translation_consistency",
                        RegionId =
                            region.Id,
                        RuleDecision =
                            "accept",
                        LayaDecision =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                    ? "accept"
                                    : "reject"
                                : null,
                        Confidence =
                            answer.Confidence,
                        Match =
                            answer.Value.HasValue
                                ? answer.Value.Value
                                : null,
                        Status =
                            answer.Value.HasValue
                                ? "shadow_only"
                                : "laya_no_answer",
                        Evidence =
                            state
                    });
            }

            progress?.Report(
                $"LayaExperimental · shadow 판단 {decisions.Count}건 완료");
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            runtimeError =
                ex.Message;

            progress?.Report(
                $"LayaExperimental 비활성 · Classic 결과 유지 · {ex.Message}");
        }

        var decisionDocument =
            new
            {
                Schema =
                    "pipeline-v2-laya-shadow-v2",
                SourceFile =
                    Path.GetFileName(
                        sourcePath),
                Track =
                    options.DecisionTrack.ToString(),
                Mode =
                    options.LayaMode.ToString(),
                Runtime =
                    runtimeDescription,
                Available =
                    runtimeError is null,
                Error =
                    runtimeError,
                Decisions =
                    decisions
            };

        var byKind =
            decisions
                .GroupBy(x =>
                    x.Kind,
                    StringComparer.Ordinal)
                .ToDictionary(
                    x =>
                        x.Key,
                    x =>
                        new
                        {
                            Compared =
                                x.Count(),
                            Matched =
                                x.Count(y =>
                                    y.Match == true),
                            Mismatched =
                                x.Count(y =>
                                    y.Match == false),
                            NoAnswer =
                                x.Count(y =>
                                    y.Match is null),
                            MeanConfidence =
                                x.Where(y =>
                                        y.Confidence.HasValue)
                                    .Select(y =>
                                        y.Confidence!.Value)
                                    .DefaultIfEmpty(0)
                                    .Average()
                        },
                    StringComparer.Ordinal);

        var summaryDocument =
            new
            {
                Schema =
                    "pipeline-v2-laya-summary-v2",
                SourceFile =
                    Path.GetFileName(
                        sourcePath),
                Track =
                    options.DecisionTrack.ToString(),
                Mode =
                    options.LayaMode.ToString(),
                Runtime =
                    runtimeDescription,
                Available =
                    runtimeError is null,
                Error =
                    runtimeError,
                TotalDecisions =
                    decisions.Count,
                Matched =
                    decisions.Count(x =>
                        x.Match == true),
                Mismatched =
                    decisions.Count(x =>
                        x.Match == false),
                NoAnswer =
                    decisions.Count(x =>
                        x.Match is null),
                ByKind =
                    byKind
            };

        var jsonOptions =
            new JsonSerializerOptions
            {
                WriteIndented =
                    true
            };

        await File.WriteAllTextAsync(
            decisionPath,
            JsonSerializer.Serialize(
                decisionDocument,
                jsonOptions),
            token);

        await File.WriteAllTextAsync(
            summaryPath,
            JsonSerializer.Serialize(
                summaryDocument,
                jsonOptions),
            token);

        if (runtimeError is not null)
        {
            return LayaShadowAuditResult.Empty;
        }

        return new LayaShadowAuditResult(
            detectorRetryRequests,
            renderRecoveryRequests);
    }

    async Task EnsureWorkerAsync(
        IProgress<string>? progress,
        CancellationToken token)
    {
        if (worker is not null &&
            !worker.HasExited &&
            workerInput is not null &&
            workerOutput is not null)
        {
            return;
        }

        await startGate.WaitAsync(token);

        try
        {
            if (worker is not null &&
                !worker.HasExited &&
                workerInput is not null &&
                workerOutput is not null)
            {
                return;
            }

            var python =
                await ResolvePythonRuntimeAsync(
                    progress,
                    token);

            if (!packageChecked)
            {
                var check =
                    await RunPythonAsync(
                        python,
                        [
                            "-c",
                            $"import laya,sys; v=getattr(laya,'__version__','unknown'); print(v); sys.exit(0 if v=='{LayaPackageVersion}' else 3)"
                        ],
                        token);

                if (check.ExitCode != 0)
                {
                    progress?.Report(
                        $"Laya {LayaPackageVersion} 설치/버전 고정 중 · 최초 1회");

                    var install =
                        await RunPythonAsync(
                            python,
                            [
                                "-m",
                                "pip",
                                "install",
                                $"laya=={LayaPackageVersion}",
                                "--disable-pip-version-check"
                            ],
                            token);

                    if (install.ExitCode != 0)
                    {
                        throw new InvalidOperationException(
                            "Laya Python 패키지 설치 실패 · " +
                            Compact(
                                string.IsNullOrWhiteSpace(
                                    install.StdErr)
                                    ? install.StdOut
                                    : install.StdErr));
                    }
                }
                else
                {
                    progress?.Report(
                        $"Laya {LayaPackageVersion} 패키지 확인 완료");
                }

                packageChecked =
                    true;
            }

            string bridgePath =
                Path.Combine(
                    AppContext.BaseDirectory,
                    "tools",
                    "laya_bridge.py");

            if (!File.Exists(
                    bridgePath))
            {
                throw new FileNotFoundException(
                    "Laya bridge script가 없습니다.",
                    bridgePath);
            }

            var start =
                new ProcessStartInfo
                {
                    FileName =
                        python.FileName,
                    UseShellExecute =
                        false,
                    RedirectStandardInput =
                        true,
                    RedirectStandardOutput =
                        true,
                    RedirectStandardError =
                        true,
                    CreateNoWindow =
                        true,
                    StandardOutputEncoding =
                        Encoding.UTF8,
                    StandardErrorEncoding =
                        Encoding.UTF8
                };

            foreach (string prefixArgument in
                     python.PrefixArguments)
            {
                start.ArgumentList.Add(
                    prefixArgument);
            }

            ConfigurePythonEnvironment(
                start);

            start.ArgumentList.Add(
                "-u");

            start.ArgumentList.Add(
                bridgePath);

            worker =
                new Process
                {
                    StartInfo =
                        start,
                    EnableRaisingEvents =
                        true
                };

            if (!worker.Start())
            {
                throw new InvalidOperationException(
                    "Laya worker를 시작할 수 없습니다.");
            }

            workerInput =
                worker.StandardInput;

            workerInput.AutoFlush =
                true;

            workerOutput =
                worker.StandardOutput;

            workerErrorDrain =
                worker.StandardError
                    .ReadToEndAsync();

            progress?.Report(
                "Laya typed-decisions checkpoint 로딩 · 최초 실행은 다운로드 시간이 걸릴 수 있음");

            string? readyLine =
                await workerOutput
                    .ReadLineAsync(token);

            if (string.IsNullOrWhiteSpace(
                    readyLine))
            {
                throw new InvalidOperationException(
                    "Laya worker 준비 응답이 없습니다.");
            }

            using var ready =
                JsonDocument.Parse(
                    readyLine);

            if (!ready.RootElement
                    .TryGetProperty(
                        "ready",
                        out var readyValue) ||
                !readyValue.GetBoolean())
            {
                string error =
                    ready.RootElement
                        .TryGetProperty(
                            "error",
                            out var errorValue)
                        ? errorValue.GetString() ??
                          "unknown"
                        : "unknown";

                throw new InvalidOperationException(
                    $"Laya checkpoint 로딩 실패 · {error}");
            }

            string version =
                ready.RootElement
                    .TryGetProperty(
                        "version",
                        out var versionValue)
                    ? versionValue.GetString() ??
                      "unknown"
                    : "unknown";

            string device =
                ready.RootElement
                    .TryGetProperty(
                        "device",
                        out var deviceValue)
                    ? deviceValue.GetString() ??
                      "unknown"
                    : "unknown";

            runtimeDescription =
                $"{python.DisplayName} · laya {version} · typed-decisions · {device}";

            progress?.Report(
                $"LayaExperimental 준비 완료 · {runtimeDescription}");
        }
        finally
        {
            startGate.Release();
        }
    }

    async Task<JsonElement?> PredictAsync(
        object state,
        IReadOnlyDictionary<string, object> questions,
        CancellationToken token)
    {
        if (worker is null ||
            worker.HasExited ||
            workerInput is null ||
            workerOutput is null)
        {
            throw new InvalidOperationException(
                "Laya worker가 준비되지 않았습니다.");
        }

        await ioGate.WaitAsync(token);

        try
        {
            string id =
                Guid.NewGuid()
                    .ToString("N");

            string request =
                JsonSerializer.Serialize(
                    new
                    {
                        id,
                        state,
                        questions
                    });

            await workerInput
                .WriteLineAsync(
                    request.AsMemory(),
                    token);

            string? responseLine =
                await workerOutput
                    .ReadLineAsync(token);

            if (string.IsNullOrWhiteSpace(
                    responseLine))
            {
                throw new InvalidOperationException(
                    "Laya worker 응답이 비어 있습니다.");
            }

            using var response =
                JsonDocument.Parse(
                    responseLine);

            if (!response.RootElement
                    .TryGetProperty(
                        "ok",
                        out var ok) ||
                !ok.GetBoolean())
            {
                string error =
                    response.RootElement
                        .TryGetProperty(
                            "error",
                            out var errorValue)
                        ? errorValue.GetString() ??
                          "unknown"
                        : "unknown";

                throw new InvalidOperationException(
                    $"Laya inference 실패 · {error}");
            }

            if (!response.RootElement
                    .TryGetProperty(
                        "result",
                        out var result))
            {
                return null;
            }

            return result.Clone();
        }
        finally
        {
            ioGate.Release();
        }
    }

    static LayaAnswer<string> ReadChoice(
        JsonElement? result,
        string questionId)
    {
        if (!TryGetAnswer(
                result,
                questionId,
                out var answer))
        {
            return new(
                null,
                null);
        }

        string? value =
            answer.TryGetProperty(
                "choice",
                out var choice)
                ? choice.GetString()
                : null;

        double? confidence =
            answer.TryGetProperty(
                "confidence",
                out var conf) &&
            conf.TryGetDouble(
                out double parsed)
                ? parsed
                : null;

        return new(
            value,
            confidence);
    }

    static LayaAnswer<bool?> ReadBoolean(
        JsonElement? result,
        string questionId)
    {
        if (!TryGetAnswer(
                result,
                questionId,
                out var answer))
        {
            return new(
                null,
                null);
        }

        bool? value =
            answer.TryGetProperty(
                "noul",
                out var noul) &&
            noul.TryGetDouble(
                out double probability)
                ? probability >= 0.5
                : null;

        double? confidence =
            answer.TryGetProperty(
                "confidence",
                out var conf) &&
            conf.TryGetDouble(
                out double parsed)
                ? parsed
                : null;

        return new(
            value,
            confidence);
    }

    static bool TryGetAnswer(
        JsonElement? result,
        string questionId,
        out JsonElement answer)
    {
        answer =
            default;

        if (!result.HasValue ||
            !result.Value.TryGetProperty(
                "answers",
                out var answers) ||
            !answers.TryGetProperty(
                questionId,
                out answer))
        {
            return false;
        }

        return true;
    }

    static List<LayaOwnershipCandidate> BuildOwnershipCandidates(
        OcrTextBlock block,
        IReadOnlyList<PageRegion> regions)
    {
        double bx1 =
            block.X;

        double by1 =
            block.Y;

        double bx2 =
            block.X +
            Math.Max(
                1,
                block.W);

        double by2 =
            block.Y +
            Math.Max(
                1,
                block.H);

        double blockArea =
            Math.Max(
                1,
                (bx2 - bx1) *
                (by2 - by1));

        double cx =
            (bx1 + bx2) /
            2.0;

        double cy =
            (by1 + by2) /
            2.0;

        return regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.TextBubble)
            .Select(text =>
            {
                double tx1 =
                    text.Bounds.Left;

                double ty1 =
                    text.Bounds.Top;

                double tx2 =
                    text.Bounds.Right;

                double ty2 =
                    text.Bounds.Bottom;

                double ix =
                    Math.Max(
                        0,
                        Math.Min(
                            bx2,
                            tx2) -
                        Math.Max(
                            bx1,
                            tx1));

                double iy =
                    Math.Max(
                        0,
                        Math.Min(
                            by2,
                            ty2) -
                        Math.Max(
                            by1,
                            ty1));

                double coverage =
                    ix *
                    iy /
                    blockArea;

                bool centerInside =
                    cx >= tx1 &&
                    cx <= tx2 &&
                    cy >= ty1 &&
                    cy <= ty2;

                double tcx =
                    (tx1 + tx2) /
                    2.0;

                double tcy =
                    (ty1 + ty2) /
                    2.0;

                double distance =
                    Math.Sqrt(
                        Math.Pow(
                            cx - tcx,
                            2) +
                        Math.Pow(
                            cy - tcy,
                            2));

                string? bubble =
                    FindParentBubbleId(
                        text,
                        regions);

                double affinity =
                    coverage *
                    4.0 +
                    (centerInside
                        ? 3.0
                        : 0.0) +
                    text.Score *
                    0.5 -
                    Math.Min(
                        2.0,
                        distance /
                        500.0);

                return new LayaOwnershipCandidate(
                    text.RegionId,
                    bubble,
                    text.Score,
                    coverage,
                    centerInside,
                    distance,
                    affinity);
            })
            .Where(x =>
                x.BlockCoverage >=
                    0.05 ||
                x.CenterInside ||
                x.CenterDistance <=
                    90)
            .OrderByDescending(x =>
                x.Affinity)
            .ThenByDescending(x =>
                x.DetectorScore)
            .Take(4)
            .ToList();
    }

    static string? FindParentBubbleId(
        PageRegion text,
        IReadOnlyList<PageRegion> regions)
    {
        double cx =
            text.Bounds.X +
            text.Bounds.Width /
            2.0;

        double cy =
            text.Bounds.Y +
            text.Bounds.Height /
            2.0;

        return regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.Bubble)
            .Select(x =>
            {
                double ix =
                    Math.Max(
                        0,
                        Math.Min(
                            text.Bounds.Right,
                            x.Bounds.Right) -
                        Math.Max(
                            text.Bounds.Left,
                            x.Bounds.Left));

                double iy =
                    Math.Max(
                        0,
                        Math.Min(
                            text.Bounds.Bottom,
                            x.Bounds.Bottom) -
                        Math.Max(
                            text.Bounds.Top,
                            x.Bounds.Top));

                double area =
                    Math.Max(
                        1,
                        text.Bounds.Width *
                        (double)text.Bounds.Height);

                double coverage =
                    ix *
                    iy /
                    area;

                bool center =
                    cx >= x.Bounds.Left &&
                    cx <= x.Bounds.Right &&
                    cy >= x.Bounds.Top &&
                    cy <= x.Bounds.Bottom;

                return new
                {
                    Region =
                        x,
                    Coverage =
                        coverage,
                    Center =
                        center
                };
            })
            .Where(x =>
                x.Center ||
                x.Coverage >=
                    0.45)
            .OrderByDescending(x =>
                x.Center)
            .ThenByDescending(x =>
                x.Coverage)
            .ThenByDescending(x =>
                x.Region.Score)
            .Select(x =>
                x.Region.RegionId)
            .FirstOrDefault();
    }

    static double AverageConfidence(
        OcrTextBlock block)
        => block.Lines.Count == 0
            ? 0
            : block.Lines.Average(x =>
                Math.Clamp(
                    x.Confidence,
                    0,
                    1));

    async Task<LayaPythonRuntime> ResolvePythonRuntimeAsync(
        IProgress<string>? progress,
        CancellationToken token)
    {
        if (pythonRuntime is not null)
        {
            return pythonRuntime;
        }

        var attempted =
            new List<string>();

        foreach (var candidate in
                 BuildPythonCandidates())
        {
            token.ThrowIfCancellationRequested();

            var probe =
                await RunPythonAsync(
                    candidate,
                    [
                        "-c",
                        "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'); raise SystemExit(0 if sys.version_info >= (3,10) else 4)"
                    ],
                    token);

            if (probe.ExitCode == 0)
            {
                string version =
                    Compact(
                        probe.StdOut);

                pythonRuntime =
                    candidate with
                    {
                        DisplayName =
                            string.IsNullOrWhiteSpace(
                                version)
                                ? candidate.DisplayName
                                : $"{candidate.DisplayName} · {version}"
                    };

                progress?.Report(
                    $"Laya Python 확인 · {pythonRuntime.DisplayName}");

                return pythonRuntime;
            }

            attempted.Add(
                candidate.DisplayName);
        }

        throw new LayaPythonNotFoundException(
            "Laya용 Python 3.10+ 실행 환경을 찾지 못했습니다. " +
            "[선택 모델 확인/설치]에서 Python 3.12를 자동 설치할 수 있습니다. " +
            $"확인 경로: {string.Join(", ", attempted)}");
    }

    static Task<ProcessResult> RunPythonAsync(
        LayaPythonRuntime runtime,
        IReadOnlyList<string> arguments,
        CancellationToken token)
    {
        var combined =
            runtime.PrefixArguments
                .Concat(
                    arguments)
                .ToArray();

        return RunProcessAsync(
            runtime.FileName,
            combined,
            token);
    }

    static async Task<ProcessResult> RunProcessAsync(
        string fileName,
        IReadOnlyList<string> arguments,
        CancellationToken token)
    {
        var start =
            new ProcessStartInfo
            {
                FileName =
                    fileName,
                UseShellExecute =
                    false,
                RedirectStandardOutput =
                    true,
                RedirectStandardError =
                    true,
                CreateNoWindow =
                    true,
                StandardOutputEncoding =
                    Encoding.UTF8,
                StandardErrorEncoding =
                    Encoding.UTF8
            };

        ConfigurePythonEnvironment(
            start);

        foreach (string argument in arguments)
        {
            start.ArgumentList.Add(
                argument);
        }

        using var process =
            new Process
            {
                StartInfo =
                    start
            };

        try
        {
            if (!process.Start())
            {
                return new ProcessResult(
                    -1,
                    "",
                    "process start failed");
            }
        }
        catch (Exception ex)
        {
            return new ProcessResult(
                -1,
                "",
                ex.Message);
        }

        Task<string> stdOut =
            process.StandardOutput
                .ReadToEndAsync();

        Task<string> stdErr =
            process.StandardError
                .ReadToEndAsync();

        await process
            .WaitForExitAsync(token);

        return new ProcessResult(
            process.ExitCode,
            await stdOut,
            await stdErr);
    }

    static string Compact(
        string? text)
    {
        if (string.IsNullOrWhiteSpace(
                text))
        {
            return "";
        }

        string oneLine =
            text
                .Replace(
                    '\r',
                    ' ')
                .Replace(
                    '\n',
                    ' ')
                .Trim();

        return oneLine.Length <=
               320
            ? oneLine
            : oneLine[..320] +
              "...";
    }

    sealed record LayaV2DuplicateGroup(
        string PhysicalOwnerKey,
        IReadOnlyList<VisionTranslation> Candidates);

    sealed class LayaShadowDecision
    {
        public string Kind { get; init; } = "";
        public int? RegionId { get; init; }
        public string RuleDecision { get; init; } = "";
        public string? LayaDecision { get; init; }
        public double? Confidence { get; init; }
        public bool? Match { get; init; }
        public string Status { get; init; } = "";
        public object? Evidence { get; init; }
    }

    sealed record LayaOwnershipCandidate(
        string TextRegionId,
        string? BubbleRegionId,
        float DetectorScore,
        double BlockCoverage,
        bool CenterInside,
        double CenterDistance,
        double Affinity);

    readonly record struct LayaAnswer<T>(
        T? Value,
        double? Confidence);

    readonly record struct ProcessResult(
        int ExitCode,
        string StdOut,
        string StdErr);
}
