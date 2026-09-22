using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

/// <summary>
/// OCR branch of Architecture 2.0.
/// RT-DETR owns page regions. RapidOCR and Baberu remain independent readers;
/// evidence is attached to translation units only after region association.
/// </summary>
public sealed class OcrPipelineService
{
    readonly OcrEngine ocr;
    readonly BaberuOcrEngine baberu;
    readonly OcrBlockGrouper blockGrouper = new();
    readonly OcrContainerUnitBuilder unitBuilder = new();

    public OcrPipelineService(
        OcrEngine ocr,
        BaberuOcrEngine baberu)
    {
        this.ocr = ocr;
        this.baberu = baberu;
    }

    public async Task<OcrStageResult> AnalyzeAsync(
        string sourcePath,
        PageAnalysisResult pageAnalysis,
        PipelineOptions options,
        CancellationToken token = default,
        IProgress<PipelineProgress>? progress = null)
    {
        var pageCandidates =
            pageAnalysis.ContainerCandidates.ToList();

        var candidateDecisions =
            pageCandidates
                .Select(
                    ContainerOcrValidator.ValidateCandidate)
                .ToList();

        var eligibleIds =
            candidateDecisions
                .Where(x => x.Eligible)
                .Select(x => x.CandidateId)
                .ToHashSet();

        var eligible =
            pageCandidates
                .Where(x =>
                    eligibleIds.Contains(
                        x.CandidateId))
                .ToList();

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.ContainerValidation,
                $"Region 컨테이너 {pageCandidates.Count}개 · focused OCR 대상 {eligible.Count}개"));

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.OcrObservation,
                "OCR A · 전체 1배·2배 / 집중 3배"));

        var observationBatch =
            await ocr.RecognizeDetailedAsync(
                sourcePath,
                token);

        RegionOcrBatch regionBatch =
            new(
                [],
                []);

        IReadOnlyList<RegionLineDecision> regionLineDecisions =
            [];

        if (options.EnableTargetedRegionOcr &&
            pageAnalysis.Regions.Any(x =>
                x.Kind is
                    PageRegionKind.TextBubble or
                    PageRegionKind.TextFree))
        {
            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.RegionOcr,
                    "RapidOCR 영역 진단 · canonical line에는 직접 합치지 않음"));

            regionBatch =
                await ocr.RecognizeRegionsAsync(
                    sourcePath,
                    pageAnalysis.Regions,
                    progress,
                    token);

            regionLineDecisions =
                TextEvidenceFusionService.Validate(
                    pageAnalysis.Regions,
                    regionBatch.Observations,
                    observationBatch.Observations,
                    token);

            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.OcrValidation,
                    $"영역 OCR 진단 · {regionBatch.Observations.Count}개 관측 / " +
                    $"evidence {regionLineDecisions.Count(x => x.Accepted)}개"));
        }

        IReadOnlyList<SecondaryOcrEvidence> secondaryEvidence =
            [];

        if (options.EnableBaberuOcr &&
            pageAnalysis.ExternalDetectorUsed &&
            baberu.IsReady)
        {
            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.RegionOcr,
                    "OCR B · Baberu 말풍선 OCR"));

            secondaryEvidence =
                await Task.Run(
                    () =>
                        baberu.Analyze(
                            sourcePath,
                            pageAnalysis,
                            token),
                    token);

            progress?.Report(
                new PipelineProgress(
                    PipelineStageKind.OcrValidation,
                    $"Baberu OCR · {secondaryEvidence.Count}개 말풍선 / " +
                    $"텍스트 evidence {secondaryEvidence.Count(x => x.Accepted)}개"));
        }

        var containerBatch =
            await ocr.RecognizeContainersAsync(
                sourcePath,
                eligible,
                progress,
                token);

        token.ThrowIfCancellationRequested();

        var lineDecisions =
            ContainerOcrValidator.ValidateLines(
                eligible,
                containerBatch.Observations,
                token);

        var acceptedContainerIds =
            lineDecisions
                .Where(x => x.Accepted)
                .Select(x => x.ObservationId)
                .ToHashSet();

        var observations =
            observationBatch.Observations
                .Concat(regionBatch.Observations)
                .Concat(containerBatch.Observations)
                .ToList();

        progress?.Report(
            new PipelineProgress(
                PipelineStageKind.OcrValidation,
                $"OCR A 컨테이너 관측 · {containerBatch.Observations.Count}개 중 " +
                $"교차 검증 {acceptedContainerIds.Count}개"));

        // Canonical OCR geometry is deliberately limited to page-wide OCR A
        // plus focused OCR A inside a confirmed RT-DETR bubble. Region OCR
        // diagnostics are not merged here; 0010's direct merge multiplied
        // observations and destabilized ownership.
        var lines =
            observationBatch.MergedLines
                .ToList();

        OcrEngine.MergeLines(
            lines,
            containerBatch.Observations
                .Where(x =>
                    acceptedContainerIds.Contains(
                        x.ObservationId))
                .Select(x => x.ToLine()));

        lines =
            lines
                .OrderBy(x => x.Y)
                .ThenBy(x => x.X)
                .ToList();

        var preliminaryBlocks =
            lines.Count == 0
                ? []
                : blockGrouper.Group(
                    lines);

        OcrUnitBuildResult unitBuild;

        if (lines.Count == 0)
        {
            unitBuild =
                new OcrUnitBuildResult(
                    [],
                    pageCandidates.Count,
                    0,
                    0);
        }
        else
        {
            unitBuild =
                await Task.Run(
                    () =>
                        unitBuilder.Build(
                            lines,
                            pageCandidates,
                            candidateDecisions,
                            containerBatch.Observations,
                            lineDecisions,
                            token),
                    token);
        }

        unitBuild =
            RecoverRtdetrTextRegionOrphans(
                unitBuild,
                pageCandidates,
                pageAnalysis.Regions);

        unitBuild =
            ConsolidateRegionOwnedUnits(
                unitBuild,
                pageCandidates,
                pageAnalysis.Regions);

        unitBuild =
            AttachRegionContainers(
                unitBuild,
                pageCandidates,
                pageAnalysis.Regions);

        unitBuild =
            ApplySecondaryEvidence(
                unitBuild,
                pageCandidates,
                secondaryEvidence);

        // ApplySecondaryEvidence can create a brand-new Baberu-only unit after
        // the first region attachment pass. Re-attach learned Bubble/TextBubble
        // evidence so those recovered units can use the same RT-DETR render
        // safety path as primary OCR units.
        unitBuild =
            AttachRegionContainers(
                unitBuild,
                pageCandidates,
                pageAnalysis.Regions);

        return new OcrStageResult(
            pageCandidates,
            observations,
            lines,
            preliminaryBlocks,
            unitBuild)
        {
            CandidateDecisions = candidateDecisions,
            ContainerAttempts = containerBatch.Attempts,
            ContainerLineDecisions = lineDecisions,
            RegionAttempts = regionBatch.Attempts,
            RegionLineDecisions = regionLineDecisions,
            SecondaryOcrEvidence = secondaryEvidence,
            PageAnalysis = pageAnalysis
        };
    }

    static OcrUnitBuildResult ConsolidateRegionOwnedUnits(
        OcrUnitBuildResult input,
        IReadOnlyList<ContainerCandidate> candidates,
        IReadOnlyList<PageRegion> regions)
    {
        if (input.Units.Count == 0 ||
            input.UnitOwnership.Count != input.Units.Count)
        {
            return input;
        }

        var candidateById =
            candidates.ToDictionary(
                x => x.CandidateId,
                StringComparer.Ordinal);

        var entries =
            input.Units
                .Select((block, index) =>
                {
                    var ownership =
                        input.UnitOwnership[index];

                    PageRegion? detectorTextRegion =
                        null;

                    if (!ownership.IsOrphan &&
                        !string.IsNullOrWhiteSpace(
                            ownership.CandidateId) &&
                        candidateById.TryGetValue(
                            ownership.CandidateId,
                            out var candidate))
                    {
                        detectorTextRegion =
                            ResolveDetectorTextRegionForBlock(
                                block,
                                regions);
                    }

                    return new
                    {
                        Block = block,
                        Ownership = ownership,
                        DetectorTextRegion =
                            detectorTextRegion
                    };
                })
                .ToList();

        var drafts =
            new List<(
                OcrTextBlock Block,
                string? CandidateId,
                IReadOnlyList<string> LineIds,
                bool IsOrphan,
                string Reason)>();

        // 0034: a broad container candidate is no longer a commit-unit
        // boundary.  Two independent RT-DETR TextBubble regions may overlap
        // the same legacy/container candidate, and consolidating only by
        // CandidateId merged adjacent balloons (page017/page028 regression).
        //
        // Consolidation is therefore allowed only inside the same immutable
        // detector text region.  If detector ownership is still unresolved,
        // keep the original unit isolated rather than guessing and merging.
        foreach (var group in
                 entries
                     .Where(x =>
                         !x.Ownership.IsOrphan &&
                         !string.IsNullOrWhiteSpace(
                             x.Ownership.CandidateId) &&
                         x.DetectorTextRegion is not null)
                     .GroupBy(
                         x =>
                             $"{x.Ownership.CandidateId}|{x.DetectorTextRegion!.RegionId}",
                         StringComparer.Ordinal))
        {
            var first =
                group.First();

            var lines =
                group
                    .SelectMany(x =>
                        x.Block.Lines)
                    .GroupBy(x =>
                        $"{Math.Round(x.X, 1)}|{Math.Round(x.Y, 1)}|" +
                        $"{Math.Round(x.W, 1)}|{Math.Round(x.H, 1)}|" +
                        $"{Normalize(x.Text)}")
                    .Select(x =>
                        x.OrderByDescending(y =>
                            y.Confidence)
                         .First())
                    .OrderBy(x =>
                        x.Y)
                    .ThenBy(x =>
                        x.X)
                    .ToList();

            if (lines.Count == 0)
                continue;

            var lineIds =
                group
                    .SelectMany(x =>
                        x.Ownership.LineIds)
                    .Distinct(
                        StringComparer.Ordinal)
                    .ToList();

            drafts.Add((
                BuildBlockFromLines(
                    -1,
                    lines),
                first.Ownership.CandidateId,
                lineIds,
                false,
                group.Count() > 1
                    ? "detector_textregion_consolidated"
                    : first.Ownership.Reason));
        }

        // Candidate-owned entries without a confident RT-DETR text-region
        // owner stay separate.  This is deliberately conservative: later
        // attachment/recovery may still bind them, but they cannot drag a
        // neighboring balloon into the same translation/commit unit.
        foreach (var entry in
                 entries.Where(x =>
                     !x.Ownership.IsOrphan &&
                     !string.IsNullOrWhiteSpace(
                         x.Ownership.CandidateId) &&
                     x.DetectorTextRegion is null))
        {
            drafts.Add((
                entry.Block,
                entry.Ownership.CandidateId,
                entry.Ownership.LineIds,
                false,
                entry.Ownership.Reason));
        }

        foreach (var entry in
                 entries.Where(x =>
                     x.Ownership.IsOrphan ||
                     string.IsNullOrWhiteSpace(
                         x.Ownership.CandidateId)))
        {
            drafts.Add((
                entry.Block,
                entry.Ownership.CandidateId,
                entry.Ownership.LineIds,
                entry.Ownership.IsOrphan,
                entry.Ownership.Reason));
        }

        var ordered =
            drafts
                .OrderBy(x =>
                    x.Block.Y)
                .ThenBy(x =>
                    x.Block.X)
                .ToList();

        var units =
            ordered
                .Select((x, id) =>
                    x.Block with
                    {
                        Id = id
                    })
                .ToList();

        var ownership =
            ordered
                .Select((x, id) =>
                    new UnitOwnershipDecision(
                        id,
                        x.CandidateId,
                        x.LineIds,
                        x.IsOrphan,
                        x.Reason))
                .ToList();

        return new OcrUnitBuildResult(
            units,
            input.ContainerCount,
            input.AssignedLineCount,
            ownership.Count(x =>
                x.IsOrphan))
        {
            LineOwnership =
                input.LineOwnership,
            UnitOwnership =
                ownership
        };
    }

    static OcrTextBlock BuildBlockFromLines(
        int id,
        IReadOnlyList<OcrLine> lines)
    {
        var ordered =
            lines
                .OrderBy(x =>
                    x.Y)
                .ThenBy(x =>
                    x.X)
                .ToList();

        double left =
            ordered.Min(x =>
                x.X);

        double top =
            ordered.Min(x =>
                x.Y);

        double right =
            ordered.Max(x =>
                x.X + x.W);

        double bottom =
            ordered.Max(x =>
                x.Y + x.H);

        string text =
            string.Join(
                " ",
                ordered
                    .Select(x =>
                        x.Text.Trim())
                    .Where(x =>
                        !string.IsNullOrWhiteSpace(
                            x)));

        string language =
            ordered
                .GroupBy(x =>
                    x.Language)
                .OrderByDescending(x =>
                    x.Count())
                .Select(x =>
                    x.Key)
                .FirstOrDefault() ??
            "en";

        return new OcrTextBlock(
            id,
            left,
            top,
            Math.Max(
                1,
                right - left),
            Math.Max(
                1,
                bottom - top),
            text,
            ordered.Count,
            language,
            ordered);
    }

    static OcrUnitBuildResult AttachRegionContainers(
        OcrUnitBuildResult input,
        IReadOnlyList<ContainerCandidate> candidates,
        IReadOnlyList<PageRegion> regions)
    {
        if (input.Units.Count == 0 ||
            input.UnitOwnership.Count != input.Units.Count)
        {
            return input;
        }

        var candidateById =
            candidates.ToDictionary(
                x => x.CandidateId,
                StringComparer.Ordinal);

        var candidateByRegionId =
            candidates
                .Where(x =>
                    !string.IsNullOrWhiteSpace(
                        x.RegionId))
                .GroupBy(
                    x => x.RegionId!,
                    StringComparer.Ordinal)
                .ToDictionary(
                    x => x.Key,
                    x => x
                        .OrderByDescending(y =>
                            y.Score)
                        .First(),
                    StringComparer.Ordinal);

        var units =
            input.Units.ToList();

        var ownership =
            input.UnitOwnership.ToList();

        var lineOwnership =
            input.LineOwnership.ToList();

        for (int i = 0;
             i < units.Count &&
             i < ownership.Count;
             i++)
        {
            var block =
                units[i];

            var owner =
                ownership[i];

            ContainerCandidate? legacyCandidate =
                null;

            if (!string.IsNullOrWhiteSpace(
                    owner.CandidateId))
            {
                candidateById.TryGetValue(
                    owner.CandidateId,
                    out legacyCandidate);
            }

            // 0035: detector text ownership is authoritative.  The legacy
            // page candidate may have grouped neighboring balloons, so choose
            // the immutable RT-DETR TextBubble from the OCR block geometry
            // first, then derive its parent Bubble and candidate.
            var detectorTextRegion =
                ResolveDetectorTextRegionForBlock(
                    block,
                    regions);

            if (detectorTextRegion is null)
            {
                if (legacyCandidate is null)
                    continue;

                units[i] =
                    block with
                    {
                        RegionId =
                            legacyCandidate.RegionId,
                        RegionContainer =
                            legacyCandidate,
                        RegionTextRegion =
                            null
                    };

                continue;
            }

            var parentBubble =
                FindParentBubble(
                    detectorTextRegion,
                    regions);

            ContainerCandidate? detectorCandidate =
                null;

            if (parentBubble is not null)
            {
                candidateByRegionId.TryGetValue(
                    parentBubble.RegionId,
                    out detectorCandidate);
            }

            var effectiveCandidate =
                detectorCandidate ??
                legacyCandidate;

            string? detectorBubbleRegionId =
                parentBubble?.RegionId ??
                effectiveCandidate?.RegionId;

            units[i] =
                block with
                {
                    RegionId =
                        detectorBubbleRegionId,
                    RegionContainer =
                        effectiveCandidate,
                    RegionTextRegion =
                        detectorTextRegion
                };

            if (detectorCandidate is null ||
                string.Equals(
                    owner.CandidateId,
                    detectorCandidate.CandidateId,
                    StringComparison.Ordinal))
            {
                continue;
            }

            ownership[i] =
                owner with
                {
                    CandidateId =
                        detectorCandidate.CandidateId,
                    IsOrphan =
                        false,
                    Reason =
                        "rtdetr_textregion_owner_authority"
                };

            foreach (string lineId in
                     owner.LineIds)
            {
                int lineIndex =
                    lineOwnership.FindIndex(x =>
                        string.Equals(
                            x.LineId,
                            lineId,
                            StringComparison.Ordinal));

                if (lineIndex < 0)
                    continue;

                var previous =
                    lineOwnership[lineIndex];

                lineOwnership[lineIndex] =
                    previous with
                    {
                        CandidateId =
                            detectorCandidate.CandidateId,
                        Assigned =
                            true,
                        Reason =
                            "rtdetr_textregion_owner_authority"
                    };
            }
        }

        return new OcrUnitBuildResult(
            units,
            input.ContainerCount,
            lineOwnership.Count(x =>
                x.Assigned),
            ownership.Count(x =>
                x.IsOrphan))
        {
            LineOwnership =
                lineOwnership,
            UnitOwnership =
                ownership
        };
    }

    static OcrUnitBuildResult RecoverRtdetrTextRegionOrphans(
        OcrUnitBuildResult input,
        IReadOnlyList<ContainerCandidate> candidates,
        IReadOnlyList<PageRegion> regions)
    {
        if (input.Units.Count == 0 ||
            input.UnitOwnership.Count != input.Units.Count ||
            candidates.Count == 0)
        {
            return input;
        }

        var candidateByRegionId =
            candidates
                .Where(x =>
                    !string.IsNullOrWhiteSpace(
                        x.RegionId))
                .GroupBy(
                    x => x.RegionId!,
                    StringComparer.Ordinal)
                .ToDictionary(
                    x => x.Key,
                    x => x
                        .OrderByDescending(y =>
                            y.Score)
                        .First(),
                    StringComparer.Ordinal);

        var units =
            input.Units.ToList();

        var ownership =
            input.UnitOwnership.ToList();

        var lineOwnership =
            input.LineOwnership.ToList();

        for (int i = 0; i < units.Count; i++)
        {
            var owner =
                ownership[i];

            if (!owner.IsOrphan ||
                !string.IsNullOrWhiteSpace(
                    owner.CandidateId))
            {
                continue;
            }

            var block =
                units[i];

            if (block.Lines.Count == 0 ||
                MeaningfulLength(
                    block.Text) == 0)
            {
                continue;
            }

            double avgConfidence =
                block.Lines.Average(
                    x => x.Confidence);

            if (avgConfidence < 0.30)
                continue;

            var textRegion =
                ResolveDetectorTextRegionForBlock(
                    block,
                    regions);

            if (textRegion is null ||
                textRegion.Score < 0.72f)
            {
                continue;
            }

            double coverage =
                Coverage(
                    ToRect(block),
                    textRegion.Bounds);

            if (coverage < 0.52 &&
                !ContainsCenter(
                    textRegion.Bounds,
                    ToRect(block)))
            {
                continue;
            }

            var parentBubble =
                FindParentBubble(
                    textRegion,
                    regions);

            if (parentBubble is null ||
                !candidateByRegionId.TryGetValue(
                    parentBubble.RegionId,
                    out var candidate))
            {
                continue;
            }

            double score =
                coverage * 4.0 +
                textRegion.Score * 2.0 +
                Math.Clamp(
                    candidate.Score,
                    0,
                    10) * 0.08;

            ownership[i] =
                owner with
                {
                    CandidateId =
                        candidate.CandidateId,
                    IsOrphan =
                        false,
                    Reason =
                        "rtdetr_textbubble_recovered"
                };

            foreach (string lineId in
                     owner.LineIds)
            {
                int lineIndex =
                    lineOwnership.FindIndex(x =>
                        string.Equals(
                            x.LineId,
                            lineId,
                            StringComparison.Ordinal));

                if (lineIndex < 0 ||
                    lineOwnership[lineIndex].Assigned)
                {
                    continue;
                }

                lineOwnership[lineIndex] =
                    new LineOwnershipDecision(
                        lineId,
                        candidate.CandidateId,
                        true,
                        "rtdetr_textbubble_recovered",
                        coverage,
                        score);
            }
        }

        return new OcrUnitBuildResult(
            units,
            input.ContainerCount,
            lineOwnership.Count(x =>
                x.Assigned),
            ownership.Count(x =>
                x.IsOrphan))
        {
            LineOwnership =
                lineOwnership,
            UnitOwnership =
                ownership
        };
    }

    public static PageRegion? ResolveDetectorTextRegionForBlock(
        OcrTextBlock block,
        IReadOnlyList<PageRegion> regions)
    {
        var blockRect =
            ToRect(
                block);

        return regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.TextBubble &&
                x.Score >= 0.65f)
            .Select(x => new
            {
                Region = x,
                BlockCoverage =
                    Coverage(
                        blockRect,
                        x.Bounds),
                CenterInside =
                    ContainsCenter(
                        x.Bounds,
                        blockRect)
            })
            .Where(x =>
                x.BlockCoverage >= 0.45 ||
                x.CenterInside)
            .OrderByDescending(x =>
                x.CenterInside)
            .ThenByDescending(x =>
                x.BlockCoverage)
            .ThenByDescending(x =>
                x.Region.Score)
            .Select(x =>
                x.Region)
            .FirstOrDefault();
    }

    static PageRegion? FindParentBubble(
        PageRegion textRegion,
        IReadOnlyList<PageRegion> regions)
    {
        return regions
            .Where(x =>
                x.Kind ==
                    PageRegionKind.Bubble)
            .Select(x => new
            {
                Region = x,
                Coverage =
                    Coverage(
                        textRegion.Bounds,
                        x.Bounds),
                CenterInside =
                    ContainsCenter(
                        x.Bounds,
                        textRegion.Bounds)
            })
            .Where(x =>
                x.CenterInside ||
                x.Coverage >= 0.45)
            .OrderByDescending(x =>
                x.CenterInside)
            .ThenByDescending(x =>
                x.Coverage)
            .ThenByDescending(x =>
                x.Region.Score)
            .Select(x =>
                x.Region)
            .FirstOrDefault();
    }

    static OpenCvSharp.Rect ToRect(
        OcrTextBlock block)
        => new(
            (int)Math.Floor(
                block.X),
            (int)Math.Floor(
                block.Y),
            Math.Max(
                1,
                (int)Math.Ceiling(
                    block.W)),
            Math.Max(
                1,
                (int)Math.Ceiling(
                    block.H)));

    static double Coverage(
        OpenCvSharp.Rect subject,
        OpenCvSharp.Rect container)
    {
        double intersection =
            IntersectionArea(
                subject,
                container);

        double area =
            Math.Max(
                1.0,
                subject.Width *
                (double)subject.Height);

        return intersection /
               area;
    }

    static bool ContainsCenter(
        OpenCvSharp.Rect container,
        OpenCvSharp.Rect subject)
    {
        double cx =
            subject.X +
            subject.Width / 2.0;

        double cy =
            subject.Y +
            subject.Height / 2.0;

        return cx >= container.Left &&
               cx <= container.Right &&
               cy >= container.Top &&
               cy <= container.Bottom;
    }

    static double IntersectionArea(
        OpenCvSharp.Rect a,
        OpenCvSharp.Rect b)
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

    static OcrUnitBuildResult ApplySecondaryEvidence(
        OcrUnitBuildResult input,
        IReadOnlyList<ContainerCandidate> candidates,
        IReadOnlyList<SecondaryOcrEvidence> evidence)
    {
        if (evidence.Count == 0)
            return input;

        var candidateById =
            candidates.ToDictionary(
                x => x.CandidateId,
                StringComparer.Ordinal);

        var evidenceByRegion =
            evidence
                .Where(x =>
                    x.Accepted &&
                    !string.IsNullOrWhiteSpace(
                        x.Text))
                .GroupBy(
                    x => x.RegionId,
                    StringComparer.Ordinal)
                .ToDictionary(
                    x => x.Key,
                    x => x
                        .OrderByDescending(y =>
                            MeaningfulLength(
                                y.Text))
                        .First(),
                    StringComparer.Ordinal);

        var units =
            input.Units.ToList();

        var unitOwnership =
            input.UnitOwnership.ToList();

        var lineOwnership =
            input.LineOwnership.ToList();

        var ownedCandidateIds =
            unitOwnership
                .Where(x =>
                    !string.IsNullOrWhiteSpace(
                        x.CandidateId))
                .Select(x => x.CandidateId!)
                .ToHashSet(
                    StringComparer.Ordinal);

        for (int i = 0;
             i < units.Count;
             i++)
        {
            if (i >= unitOwnership.Count)
                break;

            var ownership =
                unitOwnership[i];

            if (string.IsNullOrWhiteSpace(
                    ownership.CandidateId) ||
                !candidateById.TryGetValue(
                    ownership.CandidateId,
                    out var candidate))
            {
                continue;
            }

            // 0035: Baberu evidence follows the detector-authoritative Bubble
            // assigned to the block. The legacy candidate RegionId is only a
            // fallback when no detector Bubble was resolved.
            string? evidenceRegionId =
                !string.IsNullOrWhiteSpace(
                    units[i].RegionId)
                    ? units[i].RegionId
                    : candidate.RegionId;

            if (string.IsNullOrWhiteSpace(
                    evidenceRegionId) ||
                !evidenceByRegion.TryGetValue(
                    evidenceRegionId,
                    out var secondary))
            {
                continue;
            }

            string agreement =
                CompareEvidence(
                    units[i].Text,
                    secondary.Text);

            units[i] =
                units[i] with
                {
                    SecondaryOcrText =
                        secondary.Text,
                    SecondaryOcrSource =
                        secondary.Source,
                    SecondaryOcrAgreement =
                        agreement
                };
        }

        // Recovery path: RT-DETR confirms a bubble and a text region, Baberu
        // reads it, but OCR A produced no canonical unit for that container.
        // Use the learned text-region bounds as geometry instead of the whole
        // bubble so erase safety still works on a local text area.
        foreach (var secondary in
                 evidenceByRegion.Values)
        {
            var candidate =
                candidates.FirstOrDefault(x =>
                    string.Equals(
                        x.RegionId,
                        secondary.RegionId,
                        StringComparison.Ordinal));

            if (candidate is null ||
                ownedCandidateIds.Contains(
                    candidate.CandidateId))
            {
                continue;
            }

            var line =
                new OcrLine(
                    secondary.Bounds.X,
                    secondary.Bounds.Y,
                    Math.Max(
                        1,
                        secondary.Bounds.Width),
                    Math.Max(
                        1,
                        secondary.Bounds.Height),
                    secondary.Text,
                    0.82f,
                    "multi");

            int unitId =
                units.Count;

            string lineId =
                $"B{unitId + 1:0000}";

            units.Add(
                new OcrTextBlock(
                    unitId,
                    line.X,
                    line.Y,
                    line.W,
                    line.H,
                    line.Text,
                    1,
                    line.Language,
                    [line])
                {
                    SecondaryOcrText =
                        secondary.Text,
                    SecondaryOcrSource =
                        secondary.Source,
                    SecondaryOcrAgreement =
                        "secondary_only",
                    RegionId =
                        candidate.RegionId,
                    RegionContainer =
                        candidate
                });

            unitOwnership.Add(
                new UnitOwnershipDecision(
                    unitId,
                    candidate.CandidateId,
                    [lineId],
                    false,
                    "secondary_baberu_recovery"));

            lineOwnership.Add(
                new LineOwnershipDecision(
                    lineId,
                    candidate.CandidateId,
                    true,
                    "secondary_baberu_region",
                    1.0,
                    candidate.Score));

            ownedCandidateIds.Add(
                candidate.CandidateId);
        }

        return new OcrUnitBuildResult(
            units,
            input.ContainerCount,
            lineOwnership.Count(x => x.Assigned),
            unitOwnership.Count(x => x.IsOrphan))
        {
            LineOwnership =
                lineOwnership,
            UnitOwnership =
                unitOwnership
        };
    }

    static string CompareEvidence(
        string primary,
        string secondary)
    {
        string a =
            Normalize(
                primary);

        string b =
            Normalize(
                secondary);

        if (a.Length == 0)
            return "secondary_only";

        if (b.Length == 0)
            return "primary_only";

        if (string.Equals(
                a,
                b,
                StringComparison.Ordinal))
        {
            return "agree";
        }

        if (a.Length >= 4 &&
            b.Length >= 4 &&
            (a.Contains(
                 b,
                 StringComparison.Ordinal) ||
             b.Contains(
                 a,
                 StringComparison.Ordinal)))
        {
            return "partial";
        }

        return "disagree";
    }

    static int MeaningfulLength(
        string? text)
        => string.IsNullOrWhiteSpace(
               text)
            ? 0
            : text.Count(
                char.IsLetterOrDigit);

    static string Normalize(
        string? text)
        => string.IsNullOrWhiteSpace(
               text)
            ? ""
            : new string(
                text
                    .Where(
                        char.IsLetterOrDigit)
                    .Select(
                        char.ToUpperInvariant)
                    .ToArray());

    public static string FormatPassSummary(
        IReadOnlyList<OcrObservation> observations)
    {
        var parts =
            observations
                .GroupBy(x => x.Pass)
                .OrderBy(x => x.Key)
                .Select(x =>
                    $"{PassLabel(x.Key)} {x.Count()}")
                .ToList();

        return parts.Count == 0
            ? "관측 0"
            : string.Join(
                " / ",
                parts);
    }

    static string PassLabel(
        OcrPassKind pass)
        => pass switch
        {
            OcrPassKind.Global1x => "전체1x",
            OcrPassKind.Global2x => "전체2x",
            OcrPassKind.Focus3x => "focus3x",
            OcrPassKind.Region1x => "region1x",
            OcrPassKind.Region2x => "region2x",
            OcrPassKind.BaberuBubble => "baberu",
            OcrPassKind.Container1x => "container1x",
            OcrPassKind.Container2x => "container2x",
            OcrPassKind.Container3x => "container3x",
            _ => pass.ToString()
        };
}
