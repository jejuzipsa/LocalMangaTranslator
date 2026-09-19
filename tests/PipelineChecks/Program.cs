using System.IO;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.Services;
using OpenCvSharp;

var candidate = new ContainerCandidate("PC001", ContainerCandidateKind.Speech,
    new Rect(100, 200, 100, 80), "test", false, 5, 0.9, 0,
    Enumerable.Repeat((byte)255, 8000).ToArray(), 100, 80);
OcrObservation Line(string id, OcrPassKind pass, string text = "HELLO")
    => new(id, pass, 120, 220, 40, 12, text, 0.9f, "en", pass == OcrPassKind.Container1x ? 1 : 2, "PC001");
var a = Line("CO0001", OcrPassKind.Container1x);
var b = Line("CO0002", OcrPassKind.Container2x);
int count = 0;
void Check(string name, bool result)
{
    if (!result) throw new Exception(name);
    Console.WriteLine($"PASS {name}");
    count++;
}
bool AllAccepted(params OcrObservation[] lines)
    => ContainerOcrValidator.ValidateLines([candidate], lines).All(x => x.Accepted);
bool NoneAccepted(params OcrObservation[] lines)
    => ContainerOcrValidator.ValidateLines([candidate], lines).All(x => !x.Accepted);

Check("geometry eligible without existing OCR", ContainerOcrValidator.ValidateCandidate(candidate).Eligible);
Check("bad mask rejected", !ContainerOcrValidator.ValidateCandidate(candidate with { Mask = [255] }).Eligible);
Check("page border rejected", !ContainerOcrValidator.ValidateCandidate(candidate with { BorderTouches = 1 }).Eligible);
Check("low score rejected", !ContainerOcrValidator.ValidateCandidate(candidate with { Score = 3 }).Eligible);
Check("NaN score rejected", !ContainerOcrValidator.ValidateCandidate(candidate with { Score = double.NaN }).Eligible);
Check("two scales recover previously missed text", AllAccepted(a, b));
Check("punctuation normalization", AllAccepted(a, b with { Text = "Hello!" }));
Check("single pass insufficient", NoneAccepted(a));
Check("same pass repetition insufficient", NoneAccepted(a, b with { Pass = OcrPassKind.Container1x }));
Check("different text cannot corroborate", NoneAccepted(a, b with { Text = "WORLD" }));
Check("neighboring candidate cannot corroborate", NoneAccepted(a, b with { SourceKey = "PC002" }));
Check("different position cannot corroborate", NoneAccepted(a, b with { Y = 250 }));
Check("fragment inside larger box insufficient", NoneAccepted(a, b with { W = 10 }));
Check("low confidence cannot corroborate", NoneAccepted(a, b with { Confidence = 0.3f }));
Check("nonfinite geometry rejected", NoneAccepted(a with { X = double.NaN }, b));
Check("outside page-space candidate rejected", NoneAccepted(a with { X = 20 }, b with { X = 20 }));
Check("partially outside bounds rejected", NoneAccepted(a with { X = 90 }, b with { X = 90 }));
Check("single letter artwork fragments held", NoneAccepted(a with { Text = "I" }, b with { Text = "I" }));
var mask = candidate.Mask.ToArray();
for (int y = 20; y < 32; y++) for (int x = 20; x < 60; x++) mask[y * 100 + x] = 0;
Check("bounding box alone does not prove interior", ContainerOcrValidator.ValidateLines(
    [candidate with { Mask = mask }], [a, b]).All(x => !x.Accepted));
Check("page origin offsets respected", ContainerOcrValidator.MaskCoverage(candidate, a) == 1);

var unitBuilder = new OcrContainerUnitBuilder();
var eligibleA = new ContainerOcrDecision("PC001", true, "eligible_for_ocr");
var secondCandidate = candidate with
{
    CandidateId = "PC002",
    Bounds = new Rect(160, 200, 100, 80)
};
var eligibleB = new ContainerOcrDecision("PC002", true, "eligible_for_ocr");

var ownedLine1 = new OcrLine(120, 220, 40, 12, "HELLO", 0.92f, "en");
var ownedLine2 = new OcrLine(122, 240, 38, 12, "THERE", 0.91f, "en");
var oneContainer = unitBuilder.Build(
    [ownedLine1, ownedLine2],
    [candidate],
    [eligibleA],
    [],
    []);
Check("same container forms one unit", oneContainer.Units.Count == 1);
Check("each line has one explicit owner", oneContainer.LineOwnership.Count == 2 &&
    oneContainer.LineOwnership.All(x => x.Assigned && x.CandidateId == "PC001"));
Check("container unit keeps both canonical line ids", oneContainer.UnitOwnership.Count == 1 &&
    oneContainer.UnitOwnership[0].LineIds.Count == 2 &&
    oneContainer.UnitOwnership[0].CandidateId == "PC001");

var outsideLine = new OcrLine(20, 20, 30, 10, "OUTSIDE", 0.9f, "en");
var orphanResult = unitBuilder.Build(
    [outsideLine],
    [candidate],
    [eligibleA],
    [],
    []);
Check("unowned line remains orphan", orphanResult.LineOwnership.Count == 1 &&
    !orphanResult.LineOwnership[0].Assigned &&
    orphanResult.LineOwnership[0].Reason == "no_container_owner" &&
    orphanResult.OrphanGroupCount == 1);

var ambiguousLine = new OcrLine(175, 220, 20, 12, "HELLO", 0.9f, "en");
var ambiguousResult = unitBuilder.Build(
    [ambiguousLine],
    [candidate, secondCandidate],
    [eligibleA, eligibleB],
    [],
    []);
Check("ambiguous ownership is not forced", ambiguousResult.LineOwnership.Count == 1 &&
    !ambiguousResult.LineOwnership[0].Assigned &&
    ambiguousResult.LineOwnership[0].Reason == "ambiguous_container_owner");

var ownershipEvidence = new OcrObservation(
    "CO-EVIDENCE",
    OcrPassKind.Container1x,
    175, 220, 20, 12,
    "HELLO",
    0.95f,
    "en",
    1,
    "PC001");
var evidenceResult = unitBuilder.Build(
    [ambiguousLine],
    [candidate, secondCandidate],
    [eligibleA, eligibleB],
    [ownershipEvidence],
    [new ContainerLineDecision("CO-EVIDENCE", "PC001", true, "cross_scale_agreement", 1)]);
Check("validated container OCR pins ownership", evidenceResult.LineOwnership.Count == 1 &&
    evidenceResult.LineOwnership[0].Assigned &&
    evidenceResult.LineOwnership[0].CandidateId == "PC001" &&
    evidenceResult.LineOwnership[0].Reason == "container_ocr_evidence");

var duplicateCandidate = candidate with
{
    CandidateId = "PC003",
    Bounds = new Rect(102, 202, 100, 80),
    Score = 4.5
};
var duplicateResult = unitBuilder.Build(
    [ownedLine1],
    [candidate, duplicateCandidate],
    [eligibleA, new ContainerOcrDecision("PC003", true, "eligible_for_ocr")],
    [],
    []);
Check("duplicate physical containers canonicalize once", duplicateResult.ContainerCount == 1 &&
    duplicateResult.Units.Count == 1);

var weakOverlapLine = new OcrLine(185, 220, 30, 12, "EDGE", 0.90f, "en");
var weakOverlapResult = unitBuilder.Build(
    [weakOverlapLine],
    [candidate],
    [eligibleA],
    [],
    []);
Check("half overlap does not grant geometry ownership", weakOverlapResult.LineOwnership.Count == 1 &&
    !weakOverlapResult.LineOwnership[0].Assigned &&
    weakOverlapResult.LineOwnership[0].Reason == "weak_container_owner");
Check("weak ownership line stays isolated", weakOverlapResult.UnitOwnership.Count == 1 &&
    weakOverlapResult.UnitOwnership[0].IsOrphan &&
    weakOverlapResult.UnitOwnership[0].Reason == "orphan_isolated_weak_owner");

var rescueNeighbor = new OcrLine(145, 220, 40, 12, "COME", 0.93f, "en");
var rescuableLine = new OcrLine(173, 234, 30, 12, "BACK", 0.91f, "en");
var rescueResult = unitBuilder.Build(
    [rescueNeighbor, rescuableLine],
    [candidate],
    [eligibleA],
    [],
    []);
Check("weak line beside owned speech line is rescued", rescueResult.LineOwnership.Count == 2 &&
    rescueResult.LineOwnership.Any(x =>
        x.Assigned &&
        x.Reason == "rescued_neighbor_geometry"));
Check("rescued line joins container ownership", rescueResult.UnitOwnership.Any(x =>
    x.CandidateId == "PC001" &&
    x.LineIds.Count == 2));

var peerWeak1 = new OcrLine(173, 220, 30, 12, "COME", 0.92f, "en");
var peerWeak2 = new OcrLine(173, 240, 30, 12, "BACK", 0.91f, "en");
var peerRescueResult = unitBuilder.Build(
    [peerWeak1, peerWeak2],
    [candidate],
    [eligibleA],
    [],
    []);
Check("two coherent weak lines rescue each other", peerRescueResult.LineOwnership.Count == 2 &&
    peerRescueResult.LineOwnership.All(x =>
        x.Assigned &&
        x.Reason == "rescued_peer_cluster"));
Check("peer rescue forms a container unit", peerRescueResult.UnitOwnership.Count == 1 &&
    peerRescueResult.UnitOwnership[0].CandidateId == "PC001" &&
    peerRescueResult.UnitOwnership[0].LineIds.Count == 2);

var broadCandidate = candidate with
{
    CandidateId = "PC010",
    Bounds = new Rect(100, 200, 320, 320),
    Mask = Enumerable.Repeat((byte)255, 320 * 320).ToArray(),
    MaskWidth = 320,
    MaskHeight = 320,
    Score = 5.5,
    FillRatio = 0.9
};
var broadEligible = new ContainerOcrDecision("PC010", true, "eligible_for_ocr");
var farLine1 = new OcrLine(130, 230, 60, 14, "FIRST", 0.93f, "en");
var farLine2 = new OcrLine(330, 440, 60, 14, "SECOND", 0.93f, "en");
var splitResult = unitBuilder.Build(
    [farLine1, farLine2],
    [broadCandidate],
    [broadEligible],
    [],
    []);
Check("broad container splits distant text clusters", splitResult.Units.Count == 2 &&
    splitResult.UnitOwnership.All(x => x.CandidateId == "PC010") &&
    splitResult.UnitOwnership.All(x => x.Reason == "container_clustered"));
Check("split clusters keep one active owner per line", splitResult.LineOwnership.Count == 2 &&
    splitResult.LineOwnership.All(x => x.Assigned && x.CandidateId == "PC010"));

var textRegion = new PageRegion(
    "RG001",
    PageRegionKind.TextBubble,
    new Rect(110, 210, 80, 42),
    0.88f,
    "test-rtdetr");
var region1 = new OcrObservation(
    "RO0001",
    OcrPassKind.Region1x,
    120, 220, 40, 12,
    "HELLO",
    0.72f,
    "en",
    1,
    "RG001");
var region2 = region1 with
{
    ObservationId = "RO0002",
    Pass = OcrPassKind.Region2x,
    Confidence = 0.74f,
    Scale = 2
};
var regionEvidence = TextEvidenceFusionService.Validate(
    [textRegion],
    [region1, region2],
    []);
Check("independent text region cross-scale evidence accepted",
    regionEvidence.Count == 2 &&
    regionEvidence.All(x => x.Accepted && x.Reason == "region_cross_scale"));

var weakRegionOnly = TextEvidenceFusionService.Validate(
    [textRegion with { Score = 0.50f }],
    [region1 with { ObservationId = "RO0003", Confidence = 0.52f }],
    []);
Check("weak uncorroborated region OCR rejected",
    weakRegionOnly.Count == 1 &&
    !weakRegionOnly[0].Accepted);

var strongRegionOnly = TextEvidenceFusionService.Validate(
    [textRegion],
    [region1 with { ObservationId = "RO0004", Confidence = 0.88f }],
    []);
Check("strong learned text-region evidence may recover one OCR pass",
    strongRegionOnly.Count == 1 &&
    strongRegionOnly[0].Accepted &&
    strongRegionOnly[0].Reason == "strong_region_ocr");

string fusionImagePath = Path.Combine(
    Path.GetTempPath(),
    $"lmt_region_fusion_{Guid.NewGuid():N}.png");
try
{
    using var fusionImage = Mat.Zeros(400, 400, MatType.CV_8UC3).ToMat();
    Cv2.ImWrite(fusionImagePath, fusionImage);

    var learnedBubble = new PageRegion(
        "RG010",
        PageRegionKind.Bubble,
        new Rect(98, 198, 104, 84),
        0.90f,
        "test-rtdetr");

    var fusedExisting = PageAnalysisService.FuseContainers(
        [candidate],
        [learnedBubble],
        fusionImagePath);
    Check("RT-DETR bubble may reuse matching legacy mask only",
        fusedExisting.Count == 1 &&
        fusedExisting[0].DetectorMode == "rtdetr_region+legacy_mask" &&
        fusedExisting[0].RegionId == "RG010" &&
        fusedExisting[0].Score > candidate.Score);

    var newBubble = new PageRegion(
        "RG011",
        PageRegionKind.Bubble,
        new Rect(250, 250, 90, 70),
        0.92f,
        "test-rtdetr");

    var fusedFallback = PageAnalysisService.FuseContainers(
        [],
        [newBubble],
        fusionImagePath);
    Check("RT-DETR bubble can seed region container without legacy candidate",
        fusedFallback.Count == 1 &&
        fusedFallback[0].DetectorMode == "rtdetr_region_rect" &&
        fusedFallback[0].RegionId == "RG011" &&
        ContainerOcrValidator.ValidateCandidate(fusedFallback[0]).Eligible);

    var noLegacyLeak = PageAnalysisService.FuseContainers(
        [candidate],
        [newBubble],
        fusionImagePath);
    Check("unmatched legacy candidate cannot become Architecture 2.0 container",
        noLegacyLeak.Count == 1 &&
        noLegacyLeak[0].RegionId == "RG011" &&
        noLegacyLeak[0].DetectorMode == "rtdetr_region_rect");
}
finally
{
    try
    {
        if (File.Exists(fusionImagePath))
            File.Delete(fusionImagePath);
    }
    catch
    {
    }
}

var weakArtworkBlock = new OcrTextBlock(
    900,
    10,
    10,
    56,
    33,
    "LEF",
    1,
    "en",
    [new OcrLine(10, 10, 56, 33, "LEF", 0.54f, "en")]);
var hallucinatedExpansion = new VisionTranslation(
    900,
    weakArtworkBlock,
    "I DEVOUR THEM!",
    "전부 먹어치워!",
    "dialogue",
    true);
Check("low-confidence short OCR cannot expand into long dialogue",
    RenderSafetyPolicy.IsSuspiciousVisionExpansion(hallucinatedExpansion));

var highConfidenceShort = hallucinatedExpansion with
{
    Source = weakArtworkBlock with
    {
        Text = "HER",
        Lines = [new OcrLine(10, 10, 46, 27, "HER", 0.99f, "en")]
    },
    CorrectedText = "FOR MY QUEEN AND HER HEIR!"
};
Check("high-confidence short speech may use Vision context",
    !RenderSafetyPolicy.IsSuspiciousVisionExpansion(highConfidenceShort));

string webpPath = Path.Combine(
    Path.GetTempPath(),
    $"lmt_lossless_{Guid.NewGuid():N}.webp");
try
{
    using var webpSource = Mat.Zeros(3, 4, MatType.CV_8UC3).ToMat();
    webpSource.Set(0, 0, new Vec3b(3, 17, 251));
    webpSource.Set(1, 2, new Vec3b(41, 123, 219));
    webpSource.Set(2, 3, new Vec3b(255, 128, 7));

    bool wroteWebp = Cv2.ImWrite(
        webpPath,
        webpSource,
        new[]
        {
            new ImageEncodingParam(
                ImwriteFlags.WebPQuality,
                101)
        });

    Check("lossless WebP encoder writes output", wroteWebp && File.Exists(webpPath));

    using var webpRoundTrip = Cv2.ImRead(
        webpPath,
        ImreadModes.Color);

    bool exactWebp =
        !webpRoundTrip.Empty() &&
        webpRoundTrip.Rows == webpSource.Rows &&
        webpRoundTrip.Cols == webpSource.Cols &&
        webpRoundTrip.At<Vec3b>(0, 0).Equals(webpSource.At<Vec3b>(0, 0)) &&
        webpRoundTrip.At<Vec3b>(1, 2).Equals(webpSource.At<Vec3b>(1, 2)) &&
        webpRoundTrip.At<Vec3b>(2, 3).Equals(webpSource.At<Vec3b>(2, 3));

    Check("lossless WebP roundtrip preserves pixels", exactWebp);
}
finally
{
    try
    {
        if (File.Exists(webpPath))
            File.Delete(webpPath);
    }
    catch
    {
    }
}

string auditRoot = Path.Combine(
    Path.GetTempPath(),
    $"lmt_audit_{Guid.NewGuid():N}");
try
{
    OutputDirectoryLayout.Ensure(auditRoot);
    Check("output layout creates debug json and audit folders",
        Directory.Exists(OutputDirectoryLayout.Debug(auditRoot)) &&
        Directory.Exists(OutputDirectoryLayout.Json(auditRoot)) &&
        Directory.Exists(OutputDirectoryLayout.Audit(auditRoot)));

    string auditSource = Path.Combine(auditRoot, "audit_source.png");
    string auditFinal = Path.Combine(auditRoot, "audit_source.translated.webp");

    using (var src = Mat.Zeros(40, 50, MatType.CV_8UC3).ToMat())
    {
        src.SetTo(new Scalar(255, 255, 255));
        Cv2.Rectangle(src, new Rect(10, 10, 10, 8), new Scalar(0, 0, 0), -1);
        Cv2.ImWrite(auditSource, src);

        using var fin = src.Clone();
        Cv2.Rectangle(fin, new Rect(12, 12, 5, 4), new Scalar(0, 0, 255), -1);
        Cv2.ImWrite(
            auditFinal,
            fin,
            [new ImageEncodingParam(ImwriteFlags.WebPQuality, 101)]);
    }

    var emptyStage = new OcrStageResult(
        [],
        [],
        [],
        [],
        new OcrUnitBuildResult([], 0, 0, 0))
    {
        PageAnalysis = new PageAnalysisResult([], [], "test", false)
    };

    await new FinalAuditService().GenerateAsync(
        auditSource,
        auditFinal,
        auditRoot,
        emptyStage,
        []);

    Check("final audit writes compare json and summary after output",
        File.Exists(Path.Combine(
            OutputDirectoryLayout.Audit(auditRoot),
            "audit_source.final_compare.webp")) &&
        File.Exists(Path.Combine(
            OutputDirectoryLayout.Audit(auditRoot),
            "audit_source.final_audit.json")) &&
        File.Exists(Path.Combine(
            OutputDirectoryLayout.Audit(auditRoot),
            "audit_summary.json")));
}
finally
{
    try
    {
        if (Directory.Exists(auditRoot))
            Directory.Delete(auditRoot, true);
    }
    catch
    {
    }
}

if (string.Equals(
        Environment.GetEnvironmentVariable("LMT_RTDETR_DOWNLOAD_SMOKE"),
        "1",
        StringComparison.Ordinal))
{
    string? modelPath = Environment.GetEnvironmentVariable("LMT_RTDETR_MODEL_PATH");
    if (string.IsNullOrWhiteSpace(modelPath))
        throw new Exception("LMT_RTDETR_MODEL_PATH missing for download smoke");

    try
    {
        if (File.Exists(modelPath))
            File.Delete(modelPath);

        string tempPath = modelPath + ".download";
        if (File.Exists(tempPath))
            File.Delete(tempPath);

        string installed = await ExternalModelManager.EnsureRtdetrAsync();
        Check("RT-DETR first-run download finalizes model file",
            string.Equals(installed, modelPath, StringComparison.OrdinalIgnoreCase) &&
            File.Exists(modelPath) &&
            !File.Exists(tempPath) &&
            new FileInfo(modelPath).Length > 8 * 1024 * 1024);
    }
    finally
    {
        try
        {
            if (File.Exists(modelPath))
                File.Delete(modelPath);
            if (File.Exists(modelPath + ".download"))
                File.Delete(modelPath + ".download");
        }
        catch
        {
        }
    }
}

if (string.Equals(
        Environment.GetEnvironmentVariable("LMT_RTDETR_SMOKE"),
        "1",
        StringComparison.Ordinal))
{
    string detectorSmokeImage = Path.Combine(
        Path.GetTempPath(),
        $"lmt_rtdetr_smoke_{Guid.NewGuid():N}.png");

    try
    {
        using var smokeImage = Mat.Zeros(480, 320, MatType.CV_8UC3).ToMat();
        smokeImage.SetTo(new Scalar(255, 255, 255));
        Cv2.ImWrite(detectorSmokeImage, smokeImage);

        using var detector = new RtdetrPageRegionAnalyzer();
        var detectorRegions = detector.Analyze(detectorSmokeImage);

        Check("RT-DETR ONNX session and inference smoke", detectorRegions is not null);
    }
    finally
    {
        try
        {
            if (File.Exists(detectorSmokeImage))
                File.Delete(detectorSmokeImage);
        }
        catch
        {
        }
    }
}

using var cancellation = new CancellationTokenSource();
cancellation.Cancel();
bool cancelled = false;
try { ContainerOcrValidator.ValidateLines([candidate], [a, b], cancellation.Token); }
catch (OperationCanceledException) { cancelled = true; }
Check("cancellation propagates", cancelled);
Console.WriteLine($"{count} pipeline checks passed.");
