using System.IO;
using System.Reflection;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.Services;
using LocalMangaTranslator.PipelineV2.Detection;
using LocalMangaTranslator.PipelineV2.Erase;
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

using (var regionRenderSource = Mat.Zeros(400, 400, MatType.CV_8UC3).ToMat())
{
    regionRenderSource.SetTo(new Scalar(255, 255, 255));

    var regionRenderBlock = new OcrTextBlock(
        77,
        120,
        220,
        40,
        12,
        "HELLO",
        1,
        "en",
        [new OcrLine(120, 220, 40, 12, "HELLO", 0.90f, "en")])
    {
        RegionId = "RG077",
        RegionContainer = candidate with
        {
            RegionId = "RG077"
        }
    };

    var confirmedLayout = BalloonMaskService.AnalyzeConfirmedRegion(
        regionRenderSource,
        regionRenderBlock,
        "dialogue",
        regionRenderBlock.RegionContainer!);

    Check("confirmed learned region supplies render safe mask directly",
        confirmedLayout.Detected &&
        confirmedLayout.ShouldRender &&
        confirmedLayout.Mode == "region:test" &&
        confirmedLayout.SafeMask is { Length: 8000 });

    var learnedTextRegion = new PageRegion(
        "RG-TEXT-077",
        PageRegionKind.TextBubble,
        new Rect(114, 214, 58, 28),
        0.95f,
        "test-rtdetr");

    var invalidLegacyMask = regionRenderBlock.RegionContainer! with
    {
        LearnedBounds = new Rect(100, 200, 100, 80),
        Mask = [],
        MaskWidth = 0,
        MaskHeight = 0
    };

    var textRegionRescueBlock = regionRenderBlock with
    {
        RegionContainer = invalidLegacyMask,
        RegionTextRegion = learnedTextRegion
    };

    var textRegionRescueLayout = BalloonMaskService.AnalyzeConfirmedRegion(
        regionRenderSource,
        textRegionRescueBlock,
        "dialogue",
        invalidLegacyMask);

    Check("RT-DETR Bubble plus TextBubble rescues invalid legacy mask",
        textRegionRescueLayout.Detected &&
        textRegionRescueLayout.ShouldRender &&
        textRegionRescueLayout.Mode == "region:rtdetr_textbubble_rect");
}

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

var fullerLine = new OcrLine(
    108,
    220,
    84,
    16,
    "YOU'RE STILL IN THE MIDDLE",
    0.985f,
    "en");

var fragmentLine = new OcrLine(
    145,
    220,
    46,
    16,
    "THE MIDDLE",
    0.999f,
    "en");

var completeRepresentative =
    unitBuilder.Build(
        [fragmentLine, fullerLine],
        [candidate],
        [eligibleA],
        [],
        []);

Check("overlapping OCR keeps complete line instead of tiny confidence fragment",
    completeRepresentative.Units.Count == 1 &&
    completeRepresentative.Units[0].Text.Contains(
        "YOU'RE STILL IN THE MIDDLE",
        StringComparison.OrdinalIgnoreCase) &&
    !string.Equals(
        completeRepresentative.Units[0].Text.Trim(),
        "THE MIDDLE",
        StringComparison.OrdinalIgnoreCase));

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

var secondaryCandidate = candidate with
{
    CandidateId = "PC-SECONDARY",
    RegionId = "RG900",
    LearnedBounds = new Rect(100, 200, 100, 80)
};

var secondaryRecoveredBlock = new OcrTextBlock(
    901,
    120,
    220,
    40,
    12,
    "HELLO",
    1,
    "en",
    [new OcrLine(120, 220, 40, 12, "HELLO", 0.82f, "en")]);

var secondaryRecoveredBuild = new OcrUnitBuildResult(
    [secondaryRecoveredBlock],
    1,
    1,
    0)
{
    UnitOwnership =
    [
        new UnitOwnershipDecision(
            0,
            "PC-SECONDARY",
            ["B0001"],
            false,
            "secondary_baberu_recovery")
    ]
};

var secondaryBubbleRegion = new PageRegion(
    "RG900",
    PageRegionKind.Bubble,
    new Rect(100, 200, 100, 80),
    0.96f,
    "test-rtdetr");

var secondaryTextRegion = new PageRegion(
    "RG901",
    PageRegionKind.TextBubble,
    new Rect(112, 212, 76, 44),
    0.95f,
    "test-rtdetr");

var attachRegionMethod = typeof(OcrPipelineService).GetMethod(
    "AttachRegionContainers",
    BindingFlags.Static | BindingFlags.NonPublic)
    ?? throw new Exception("AttachRegionContainers reflection lookup failed");

var secondaryReattached = (OcrUnitBuildResult)(
    attachRegionMethod.Invoke(
        null,
        [
            secondaryRecoveredBuild,
            new[] { secondaryCandidate },
            new[] { secondaryBubbleRegion, secondaryTextRegion }
        ])
    ?? throw new Exception("AttachRegionContainers returned null"));

Check("Baberu recovery can receive RT-DETR TextBubble on second attachment pass",
    secondaryReattached.Units.Count == 1 &&
    secondaryReattached.Units[0].RegionContainer?.CandidateId == "PC-SECONDARY" &&
    secondaryReattached.Units[0].RegionTextRegion?.RegionId == "RG901");

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
        fusedExisting[0].LearnedBounds == learnedBubble.Bounds &&
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

    var leftBubble = new PageRegion(
        "RG012",
        PageRegionKind.Bubble,
        new Rect(80, 190, 80, 80),
        0.89f,
        "test-rtdetr");

    var rightBubble = new PageRegion(
        "RG013",
        PageRegionKind.Bubble,
        new Rect(140, 210, 80, 70),
        0.88f,
        "test-rtdetr");

    var oneLegacyTwoLearned = PageAnalysisService.FuseContainers(
        [candidate],
        [leftBubble, rightBubble],
        fusionImagePath);

    Check("one legacy contour cannot be reused by two learned bubbles",
        oneLegacyTwoLearned.Count == 2 &&
        oneLegacyTwoLearned.Count(x =>
            x.DetectorMode == "rtdetr_region+legacy_mask") == 1 &&
        oneLegacyTwoLearned.Count(x =>
            x.DetectorMode == "rtdetr_region_rect") == 1);
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

string v2MaskSourcePath = Path.Combine(
    Path.GetTempPath(),
    $"lmt_v2_mask_{Guid.NewGuid():N}.png");
try
{
    using var v2Source = Mat.Zeros(220, 420, MatType.CV_8UC3).ToMat();
    v2Source.SetTo(new Scalar(255, 255, 255));

    Cv2.PutText(
        v2Source,
        "YOU'RE STILL IN THE MIDDLE",
        new Point(55, 118),
        HersheyFonts.HersheySimplex,
        0.72,
        new Scalar(0, 0, 0),
        2,
        LineTypes.AntiAlias);

    Cv2.ImWrite(v2MaskSourcePath, v2Source);

    var v2Bubble = new PageRegion(
        "V2-B1",
        PageRegionKind.Bubble,
        new Rect(25, 55, 360, 105),
        0.97f,
        "test-rtdetr");

    var v2Text = new PageRegion(
        "V2-T1",
        PageRegionKind.TextBubble,
        new Rect(45, 82, 330, 52),
        0.96f,
        "test-rtdetr");

    var v2Snapshot = V2DetectionSnapshot.Create(
        new PageAnalysisResult(
            [v2Bubble, v2Text],
            [],
            "v2-test",
            true));

    Check("V2 snapshot preserves RT-DETR TextBubble geometry",
        v2Snapshot.TextTargets.Count == 1 &&
        v2Snapshot.TextTargets[0].TextBounds == v2Text.Bounds &&
        v2Snapshot.TextTargets[0].BubbleBounds == v2Bubble.Bounds);

    using var v2Mask = ComicTranslateComponentMask.Build(
        v2Source,
        v2Text.Bounds,
        v2Bubble.Bounds);

    Check("V2 text mask finds lettering without OCR canonical lines",
        Cv2.CountNonZero(v2Mask) > 100);

    bool outsideTouched = false;
    for (int y = 0; y < v2Mask.Rows && !outsideTouched; y++)
    {
        for (int x = 0; x < v2Mask.Cols; x++)
        {
            if (v2Mask.At<byte>(y, x) == 0)
                continue;

            if (x < v2Text.Bounds.Left ||
                x >= v2Text.Bounds.Right ||
                y < v2Text.Bounds.Top ||
                y >= v2Text.Bounds.Bottom)
            {
                outsideTouched = true;
                break;
            }
        }
    }

    Check("V2 text mask cannot escape immutable TextBubble geometry",
        !outsideTouched);
}
finally
{
    try
    {
        if (File.Exists(v2MaskSourcePath))
            File.Delete(v2MaskSourcePath);
    }
    catch
    {
    }
}

string v2DarkMaskSourcePath = Path.Combine(
    Path.GetTempPath(),
    $"lmt_v2_dark_mask_{Guid.NewGuid():N}.png");
try
{
    using var darkSource =
        Mat.Zeros(180, 360, MatType.CV_8UC3).ToMat();

    darkSource.SetTo(
        new Scalar(245, 245, 245));

    var darkBubble =
        new Rect(40, 45, 280, 90);

    var darkText =
        new Rect(60, 60, 240, 58);

    Cv2.Rectangle(
        darkSource,
        darkBubble,
        new Scalar(8, 8, 8),
        thickness: -1);

    Cv2.PutText(
        darkSource,
        "WHITE ON BLACK",
        new Point(72, 99),
        HersheyFonts.HersheySimplex,
        0.72,
        new Scalar(245, 245, 245),
        2,
        LineTypes.AntiAlias);

    Cv2.ImWrite(
        v2DarkMaskSourcePath,
        darkSource);

    using var darkMask =
        ComicTranslateComponentMask.Build(
            darkSource,
            darkText,
            darkBubble);

    int darkMaskPixels =
        Cv2.CountNonZero(darkMask);

    Check("V2 text mask supports white lettering on dark captions",
        darkMaskPixels > 100);

    Check("V2 dark text mask never degenerates into full-box erase",
        darkMaskPixels <
        darkText.Width * darkText.Height * 0.48);
}
finally
{
    try
    {
        if (File.Exists(v2DarkMaskSourcePath))
            File.Delete(v2DarkMaskSourcePath);
    }
    catch
    {
    }
}

string v2ColoredMaskSourcePath = Path.Combine(
    Path.GetTempPath(),
    $"lmt_v2_colored_mask_{Guid.NewGuid():N}.png");
try
{
    using var coloredSource =
        Mat.Zeros(190, 420, MatType.CV_8UC3).ToMat();

    coloredSource.SetTo(
        new Scalar(235, 235, 235));

    var coloredBubble =
        new Rect(40, 38, 340, 112);

    var coloredText =
        new Rect(58, 56, 304, 76);

    Cv2.Rectangle(
        coloredSource,
        coloredBubble,
        new Scalar(252, 252, 252),
        thickness: -1);

    Cv2.PutText(
        coloredSource,
        "NONONO",
        new Point(90, 112),
        HersheyFonts.HersheySimplex,
        1.45,
        new Scalar(30, 30, 220),
        4,
        LineTypes.AntiAlias);

    Cv2.ImWrite(
        v2ColoredMaskSourcePath,
        coloredSource);

    using var coloredMask =
        ComicTranslateComponentMask.Build(
            coloredSource,
            coloredText,
            coloredBubble);

    int coloredMaskPixels =
        Cv2.CountNonZero(
            coloredMask);

    Check("V2 mask rescues colored comic lettering",
        coloredMaskPixels > 250);

    Check("V2 colored rescue stays local instead of consuming detector box",
        coloredMaskPixels <
        coloredText.Width * coloredText.Height * 0.35);

    using var coloredReviewMask =
        ComicTranslateComponentMask.Build(
            coloredSource,
            coloredText,
            coloredBubble,
            includeColorRescue: false);

    Check("V2 residual review can disable chroma rescue",
        Cv2.CountNonZero(coloredReviewMask) <=
        coloredMaskPixels);
}
finally
{
    try
    {
        if (File.Exists(v2ColoredMaskSourcePath))
            File.Delete(v2ColoredMaskSourcePath);
    }
    catch
    {
    }
}

// 0029 regression: comic display lettering often uses red fill + thick
// black outline. The erase mask must select the ink, not flood-fill the white
// counter inside O/P/R or the negative space between letters.
using (var outlinedGlyphSource =
       Mat.Zeros(
           180,
           180,
           MatType.CV_8UC3)
       .ToMat())
{
    outlinedGlyphSource.SetTo(
        new Scalar(
            252,
            252,
            252));

    var outlinedBubble =
        new Rect(
            20,
            20,
            140,
            140);

    var outlinedText =
        new Rect(
            38,
            38,
            84,
            84);

    Cv2.Circle(
        outlinedGlyphSource,
        new Point(
            80,
            80),
        30,
        new Scalar(
            10,
            10,
            10),
        12,
        LineTypes.AntiAlias);

    Cv2.Circle(
        outlinedGlyphSource,
        new Point(
            80,
            80),
        30,
        new Scalar(
            30,
            30,
            220),
        6,
        LineTypes.AntiAlias);

    using var outlinedGlyphMask =
        ComicTranslateComponentMask.Build(
            outlinedGlyphSource,
            outlinedText,
            outlinedBubble);

    Check("V2 0029 outlined comic glyph keeps O counter empty",
        outlinedGlyphMask.At<byte>(
            80,
            80) == 0);

    Check("V2 0029 outlined comic glyph still selects colored/outlined ink",
        Cv2.CountNonZero(
            outlinedGlyphMask) > 250 &&
        outlinedGlyphMask.At<byte>(
            50,
            80) != 0);
}

// 0030 regression: foreground segmentation and background reconstruction
// are separate decisions. A stylized colored glyph on a flat balloon should
// restore from the balloon background instead of feeding colored ink to Telea.
using (var flatBackgroundSource =
       Mat.Zeros(
           180,
           180,
           MatType.CV_8UC3)
       .ToMat())
{
    flatBackgroundSource.SetTo(
        new Scalar(
            252,
            252,
            252));

    var flatBubble =
        new Rect(
            20,
            20,
            140,
            140);

    var flatTextBounds =
        new Rect(
            38,
            38,
            84,
            84);

    Cv2.Circle(
        flatBackgroundSource,
        new Point(
            80,
            80),
        30,
        new Scalar(
            10,
            10,
            10),
        12,
        LineTypes.AntiAlias);

    Cv2.Circle(
        flatBackgroundSource,
        new Point(
            80,
            80),
        30,
        new Scalar(
            30,
            30,
            220),
        6,
        LineTypes.AntiAlias);

    var flatTarget =
        new V2TextTarget(
            "V2-0030-FLAT",
            PageRegionKind.TextBubble,
            flatTextBounds,
            0.95f,
            "V2-0030-BUBBLE",
            flatBubble,
            0.96f);

    using var flatGlyphMask =
        ComicTranslateComponentMask.Build(
            flatBackgroundSource,
            flatTextBounds,
            flatBubble);

    var flatAudit =
        BackgroundReconstructionV2.Analyze(
            flatBackgroundSource,
            flatTarget,
            flatGlyphMask);

    Check("V2 0030 classifies dominant balloon background as flat",
        flatAudit.FlatAccepted &&
        flatAudit.Strategy == "FLAT_FILL" &&
        flatAudit.DominantMatchRatio >= 0.70);

    using var flatReconstructed =
        flatBackgroundSource.Clone();

    BackgroundReconstructionV2.ApplyFlatFill(
        flatReconstructed,
        flatGlyphMask,
        flatAudit);

    var restoredInkPixel =
        flatReconstructed.At<Vec3b>(
            50,
            80);

    Check("V2 0030 flat fill restores colored glyph ink to balloon background",
        restoredInkPixel.Item0 >= 240 &&
        restoredInkPixel.Item1 >= 240 &&
        restoredInkPixel.Item2 >= 240);

    Check("V2 0030 flat fill keeps O counter unmasked",
        flatGlyphMask.At<byte>(
            80,
            80) == 0 &&
        flatReconstructed.At<Vec3b>(
            80,
            80).Item0 >= 240);
}

// 0031 regression: background candidates immediately adjacent to a
// stylized colored glyph are not trustworthy. Excluding a halo around the
// accepted glyph mask must recover the actual flat balloon background even
// when the mask itself does not cover the full black/red display lettering.
using (var contaminatedFlatSource =
       Mat.Zeros(
           180,
           220,
           MatType.CV_8UC3)
       .ToMat())
using (var partialGlyphMask =
       Mat.Zeros(
           180,
           220,
           MatType.CV_8UC1)
       .ToMat())
{
    contaminatedFlatSource.SetTo(
        new Scalar(
            252,
            252,
            252));

    var contaminatedTextBounds =
        new Rect(
            30,
            30,
            160,
            120);

    Cv2.Circle(
        contaminatedFlatSource,
        new Point(
            110,
            88),
        38,
        new Scalar(
            8,
            8,
            8),
        14,
        LineTypes.AntiAlias);

    Cv2.Circle(
        contaminatedFlatSource,
        new Point(
            110,
            88),
        38,
        new Scalar(
            25,
            35,
            225),
        7,
        LineTypes.AntiAlias);

    Cv2.Circle(
        partialGlyphMask,
        new Point(
            110,
            88),
        38,
        Scalar.White,
        7,
        LineTypes.AntiAlias);

    var contaminatedTarget =
        new V2TextTarget(
            "V2-0031-CONTAMINATED-FLAT",
            PageRegionKind.TextBubble,
            contaminatedTextBounds,
            0.95f,
            "V2-0031-BUBBLE",
            new Rect(
                24,
                24,
                172,
                132),
            0.96f);

    var contaminatedAudit =
        BackgroundReconstructionV2.Analyze(
            contaminatedFlatSource,
            contaminatedTarget,
            partialGlyphMask);

    Check("V2 0031 excludes glyph-adjacent color contamination before background clustering",
        contaminatedAudit.FlatAccepted &&
        contaminatedAudit.Strategy == "FLAT_FILL" &&
        contaminatedAudit.BackgroundB >= 240 &&
        contaminatedAudit.BackgroundG >= 240 &&
        contaminatedAudit.BackgroundR >= 240 &&
        contaminatedAudit.ExclusionRadius >= 3 &&
        contaminatedAudit.DominantMatchRatio >= 0.60);
}

// 0032 regression: a target that was classified FLAT_FILL must keep that
// strategy during retry. Expanding the retry mask over surviving colored ink
// should repaint it from the already-estimated flat background, never use the
// colored residue itself as Telea source material.
using (var retryFlatImage =
       Mat.Zeros(
           120,
           160,
           MatType.CV_8UC3)
       .ToMat())
using (var retryFlatMask =
       Mat.Zeros(
           120,
           160,
           MatType.CV_8UC1)
       .ToMat())
{
    retryFlatImage.SetTo(
        new Scalar(
            246,
            242,
            241));

    Cv2.Rectangle(
        retryFlatImage,
        new Rect(
            55,
            42,
            50,
            28),
        new Scalar(
            35,
            45,
            225),
        thickness: -1);

    Cv2.Rectangle(
        retryFlatMask,
        new Rect(
            48,
            36,
            64,
            40),
        Scalar.White,
        thickness: -1);

    var retryFlatAudit =
        new V2BackgroundReconstructionAudit(
            "V2-0032-FLAT-RETRY",
            PageRegionKind.TextBubble.ToString(),
            new Rect(
                30,
                24,
                100,
                72),
            "FLAT_FILL",
            Cv2.CountNonZero(
                retryFlatMask),
            100,
            246,
            242,
            241,
            0.88,
            16,
            20,
            0.625,
            7,
            true,
            "dominant_purified_background_cluster");

    BackgroundReconstructionV2.ApplyFlatFill(
        retryFlatImage,
        retryFlatMask,
        retryFlatAudit);

    var retriedPixel =
        retryFlatImage.At<Vec3b>(
            58,
            78);

    Check("V2 0032 flat retry removes colored residue with stored background instead of Telea",
        retriedPixel.Item0 == 246 &&
        retriedPixel.Item1 == 242 &&
        retriedPixel.Item2 == 241);
}

using (var artworkSource =
       Mat.Zeros(
           160,
           200,
           MatType.CV_8UC3)
       .ToMat())
using (var artworkMask =
       Mat.Zeros(
           160,
           200,
           MatType.CV_8UC1)
       .ToMat())
{
    for (int y = 20; y < 140; y++)
    {
        for (int x = 20; x < 180; x++)
        {
            bool alternate =
                ((x / 12) +
                 (y / 12)) %
                2 == 0;

            artworkSource.Set(
                y,
                x,
                alternate
                    ? new Vec3b(
                        30,
                        50,
                        200)
                    : new Vec3b(
                        210,
                        180,
                        40));
        }
    }

    Cv2.Rectangle(
        artworkMask,
        new Rect(
            80,
            60,
            40,
            35),
        Scalar.White,
        thickness: -1);

    var artworkTarget =
        new V2TextTarget(
            "V2-0030-ART",
            PageRegionKind.TextFree,
            new Rect(
                20,
                20,
                160,
                120),
            0.95f,
            null,
            null,
            null);

    var artworkAudit =
        BackgroundReconstructionV2.Analyze(
            artworkSource,
            artworkTarget,
            artworkMask);

    Check("V2 0030 keeps complex artwork on Telea path",
        !artworkAudit.FlatAccepted &&
        artworkAudit.Strategy == "TELEA");
}

// 0020 regression: initial erase may use chroma rescue, but post-inpaint
// residual review must not reinterpret harmless color variation as lettering.
Check("V2 residual policy remains density based after chroma review split",
    ErasePipelineV2.IsResidualAcceptable(
        3780,
        278,
        new Rect(63, 56, 157, 75)));

Check("V2 0022 accepts ten-percent low-density post-inpaint residue",
    ErasePipelineV2.IsResidualAcceptable(
        3780,
        378,
        new Rect(63, 56, 157, 75)));

Check("V2 0022 accepts sparse caption texture without retry inflation",
    ErasePipelineV2.IsResidualAcceptable(
        7233,
        379,
        new Rect(1046, 243, 223, 64)));

Check("V2 residual review tolerates tiny post-inpaint speckles",
    ErasePipelineV2.IsResidualAcceptable(
        2303,
        10,
        new Rect(540, 193, 102, 77)));

Check("V2 residual review tolerates low-density textured caption noise",
    ErasePipelineV2.IsResidualAcceptable(
        5944,
        343,
        new Rect(179, 1187, 288, 52)));

Check("V2 residual review still rejects substantial remaining lettering",
    !ErasePipelineV2.IsResidualAcceptable(
        3537,
        587,
        new Rect(1061, 1671, 163, 52)));

Check("V2 0024 accepts inpaint texture when outside-mask residue is tiny",
    ErasePipelineV2.IsResidualAcceptable(
        3537,
        433,
        40,
        new Rect(1061, 1671, 163, 52)));

Check("V2 0024 still rejects real outside-mask lettering",
    !ErasePipelineV2.IsResidualAcceptable(
        3537,
        433,
        180,
        new Rect(1061, 1671, 163, 52)));

using (var originalMask =
       Mat.Zeros(
           20,
           20,
           MatType.CV_8UC1)
       .ToMat())
using (var residualMask =
       Mat.Zeros(
           20,
           20,
           MatType.CV_8UC1)
       .ToMat())
{
    Cv2.Rectangle(
        originalMask,
        new Rect(
            5,
            5,
            10,
            10),
        Scalar.White,
        thickness: -1);

    Cv2.Rectangle(
        residualMask,
        new Rect(
            7,
            7,
            4,
            4),
        Scalar.White,
        thickness: -1);

    Cv2.Rectangle(
        residualMask,
        new Rect(
            15,
            8,
            2,
            3),
        Scalar.White,
        thickness: -1);

    Check("V2 0024 residual review ignores re-segmented pixels inside erased glyph mask",
        ErasePipelineV2.CountResidualOutsideOriginalMask(
            residualMask,
            originalMask) == 6);
}

// 0025 regression: review must follow source-glyph persistence, not any
// high-contrast structure that happens to be near the detector text box.
using (var coreMask =
       Mat.Zeros(
           40,
           40,
           MatType.CV_8UC1)
       .ToMat())
using (var residualMask =
       Mat.Zeros(
           40,
           40,
           MatType.CV_8UC1)
       .ToMat())
{
    Cv2.Rectangle(
        coreMask,
        new Rect(
            14,
            14,
            10,
            10),
        Scalar.White,
        thickness: -1);

    // Nearby bubble/artwork edge: inside the halo but never touching the
    // source glyph core.
    Cv2.Line(
        residualMask,
        new Point(
            8,
            10),
        new Point(
            30,
            10),
        Scalar.White,
        2);

    // Actual source-glyph residue.
    Cv2.Rectangle(
        residualMask,
        new Rect(
            17,
            17,
            4,
            4),
        Scalar.White,
        thickness: -1);

    using var linked =
        ErasePipelineV2.BuildCoreLinkedResidual(
            residualMask,
            coreMask);

    Check("V2 0025 ignores nearby border components that do not touch glyph core",
        Cv2.CountNonZero(
            linked) == 16);
}

using (var sourceGlyph =
       Mat.Zeros(
           32,
           32,
           MatType.CV_8UC3)
       .ToMat())
using (var cleanedGlyph =
       Mat.Zeros(
           32,
           32,
           MatType.CV_8UC3)
       .ToMat())
using (var glyphCore =
       Mat.Zeros(
           32,
           32,
           MatType.CV_8UC1)
       .ToMat())
using (var glyphResidual =
       Mat.Zeros(
           32,
           32,
           MatType.CV_8UC1)
       .ToMat())
{
    sourceGlyph.SetTo(
        new Scalar(
            255,
            255,
            255));

    cleanedGlyph.SetTo(
        new Scalar(
            255,
            255,
            255));

    Cv2.Rectangle(
        sourceGlyph,
        new Rect(
            10,
            10,
            10,
            10),
        new Scalar(
            0,
            0,
            0),
        thickness: -1);

    Cv2.Rectangle(
        glyphCore,
        new Rect(
            10,
            10,
            10,
            10),
        Scalar.White,
        thickness: -1);

    Cv2.Rectangle(
        glyphResidual,
        new Rect(
            10,
            10,
            10,
            10),
        Scalar.White,
        thickness: -1);

    int erasedPersistence =
        ErasePipelineV2.CountOriginalGlyphPersistence(
            sourceGlyph,
            cleanedGlyph,
            glyphCore,
            glyphResidual,
            new Rect(
                8,
                8,
                14,
                14));

    Check("V2 0025 accepts a fully changed source glyph core",
        erasedPersistence == 0 &&
        ErasePipelineV2.IsCorePersistenceAcceptable(
            100,
            100,
            erasedPersistence));

    sourceGlyph.CopyTo(
        cleanedGlyph);

    int survivingPersistence =
        ErasePipelineV2.CountOriginalGlyphPersistence(
            sourceGlyph,
            cleanedGlyph,
            glyphCore,
            glyphResidual,
            new Rect(
                8,
                8,
                14,
                14));

    Check("V2 0025 rejects genuinely surviving source glyph pixels",
        survivingPersistence == 100 &&
        !ErasePipelineV2.IsCorePersistenceAcceptable(
            100,
            100,
            survivingPersistence));
}

Check("V2 0027 cleaned checkpoint requires exact detector ID set match",
    ErasePipelineV2.CheckpointIdsMatch(
        ["B001", "B002", "B003"],
        ["B003", "B001", "B002"]));

Check("V2 0027 cleaned checkpoint catches a missing bubble ID",
    !ErasePipelineV2.CheckpointIdsMatch(
        ["B001", "B002", "B003"],
        ["B001", "B003"]));

Check("V2 0027 cleaned checkpoint catches an unexpected replacement ID",
    !ErasePipelineV2.CheckpointIdsMatch(
        ["B001", "B002", "B003"],
        ["B001", "B002", "B004"]));

Check("V2 0028 cleaned verifier matches residual text inside original target",
    CleanedStateVerifier.IsResidualMatch(
        new Rect(
            100,
            100,
            200,
            80),
        new Rect(
            130,
            120,
            120,
            35)));

Check("V2 0028 cleaned verifier allows nearby text outside original target",
    !CleanedStateVerifier.IsResidualMatch(
        new Rect(
            100,
            100,
            200,
            80),
        new Rect(
            315,
            110,
            100,
            35)));

Check("V2 0028 cleaned verifier accepts a partially clipped surviving detection",
    CleanedStateVerifier.IsResidualMatch(
        new Rect(
            100,
            100,
            200,
            80),
        new Rect(
            275,
            120,
            55,
            35)));

var primaryPassDecision =
    ErasePipelineV2.ResolveReviewDecision(
        primaryClean: true,
        secondaryClean: false);

Check("V2 0026 primary pass cannot be overturned by secondary reviewer",
    primaryPassDecision.Clean &&
    !primaryPassDecision.SecondaryRan &&
    !primaryPassDecision.RescuedBySecondary &&
    primaryPassDecision.Reason ==
        "primary_residual_clean");

var secondaryRescueDecision =
    ErasePipelineV2.ResolveReviewDecision(
        primaryClean: false,
        secondaryClean: true);

Check("V2 0026 secondary glyph review only rescues primary failures",
    secondaryRescueDecision.Clean &&
    secondaryRescueDecision.SecondaryRan &&
    secondaryRescueDecision.RescuedBySecondary &&
    secondaryRescueDecision.Reason ==
        "secondary_glyph_rescue");

var doubleFailDecision =
    ErasePipelineV2.ResolveReviewDecision(
        primaryClean: false,
        secondaryClean: false);

Check("V2 0026 keeps original when both review layers fail",
    !doubleFailDecision.Clean &&
    doubleFailDecision.SecondaryRan &&
    !doubleFailDecision.RescuedBySecondary &&
    doubleFailDecision.Reason ==
        "original_glyph_persistence");

var firstPassSelection =
    ErasePipelineV2.SelectBestResidualPass(
        433,
        587,
        true);

Check("V2 0023 keeps first erase pass when retry is worse",
    firstPassSelection.SelectedPass == "first" &&
    firstPassSelection.SelectedResidualPixels == 433);

var retryPassSelection =
    ErasePipelineV2.SelectBestResidualPass(
        587,
        279,
        true);

Check("V2 0023 keeps retry erase pass when it improves residual",
    retryPassSelection.SelectedPass == "retry" &&
    retryPassSelection.SelectedResidualPixels == 279);

string v2MainRoot = Path.Combine(
    Path.GetTempPath(),
    $"lmt_v2_main_{Guid.NewGuid():N}");
try
{
    OutputDirectoryLayout.Ensure(v2MainRoot);

    string v2MainSourcePath =
        Path.Combine(v2MainRoot, "v2_main.png");

    using var v2MainSource =
        Mat.Zeros(220, 420, MatType.CV_8UC3).ToMat();

    v2MainSource.SetTo(
        new Scalar(255, 255, 255));

    Cv2.PutText(
        v2MainSource,
        "HELLO",
        new Point(70, 105),
        HersheyFonts.HersheySimplex,
        0.9,
        new Scalar(0, 0, 0),
        2,
        LineTypes.AntiAlias);

    Cv2.PutText(
        v2MainSource,
        "KEEP",
        new Point(250, 105),
        HersheyFonts.HersheySimplex,
        0.9,
        new Scalar(0, 0, 0),
        2,
        LineTypes.AntiAlias);

    Cv2.ImWrite(
        v2MainSourcePath,
        v2MainSource);

    var bubbleA = new PageRegion(
        "VB-A",
        PageRegionKind.Bubble,
        new Rect(35, 55, 155, 85),
        0.97f,
        "test-rtdetr");

    var textA = new PageRegion(
        "VT-A",
        PageRegionKind.TextBubble,
        new Rect(55, 75, 120, 48),
        0.96f,
        "test-rtdetr");

    var bubbleB = new PageRegion(
        "VB-B",
        PageRegionKind.Bubble,
        new Rect(220, 55, 165, 85),
        0.97f,
        "test-rtdetr");

    var textB = new PageRegion(
        "VT-B",
        PageRegionKind.TextBubble,
        new Rect(240, 75, 125, 48),
        0.96f,
        "test-rtdetr");

    var mainSnapshot =
        V2DetectionSnapshot.Create(
            new PageAnalysisResult(
                [bubbleA, textA, bubbleB, textB],
                [],
                "v2-main-test",
                true));

    var translatedBlock =
        new OcrTextBlock(
            2001,
            55,
            75,
            120,
            48,
            "HELLO",
            1,
            "en",
            [new OcrLine(65, 82, 90, 28, "HELLO", 0.99f, "en")])
        {
            RegionId = "VB-A",
            RegionTextRegion = textA
        };

    var translatedRegion =
        new VisionTranslation(
            2001,
            translatedBlock,
            "HELLO",
            "안녕",
            "dialogue",
            true);

    var selection =
        V2EraseSelector.Select(
            mainSnapshot,
            [translatedRegion]);

    Check("V2 selector erases only translated immutable target",
        selection.TextRegionIds.SetEquals(["VT-A"]) &&
        selection.TranslationRegionIds.SetEquals([2001]));

    Check("V2 selector keeps TextBubble for erase and parent Bubble for layout",
        selection.Bindings.TryGetValue(2001, out var mainBinding) &&
        mainBinding.TextBounds == textA.Bounds &&
        mainBinding.LayoutBounds == bubbleA.Bounds &&
        mainBinding.BubbleRegionId == "VB-A" &&
        mainBinding.LayoutMode == "rtdetr_parent_bubble" &&
        mainBinding.TextRegionIds.SequenceEqual(["VT-A"]));

    var freeTextRegion =
        new PageRegion(
            "VT-FREE",
            PageRegionKind.TextFree,
            new Rect(72, 158, 210, 34),
            0.94f,
            "test-rtdetr");

    var freeSnapshot =
        V2DetectionSnapshot.Create(
            new PageAnalysisResult(
                [freeTextRegion],
                [],
                "v2-free-test",
                true));

    Check("V2 snapshot carries RT-DETR TextFree geometry",
        freeSnapshot.TextTargets.Count == 1 &&
        freeSnapshot.TextTargets[0].Kind == PageRegionKind.TextFree &&
        freeSnapshot.TextTargets[0].TextBounds == freeTextRegion.Bounds &&
        freeSnapshot.TextTargets[0].BubbleBounds is null);

    var freeBlock =
        new OcrTextBlock(
            2002,
            74,
            160,
            205,
            30,
            "YOU'VE BEEN LOST IN THE DARK",
            1,
            "en",
            [new OcrLine(
                74,
                160,
                205,
                30,
                "YOU'VE BEEN LOST IN THE DARK",
                0.98f,
                "en")]);

    var freeTranslation =
        new VisionTranslation(
            2002,
            freeBlock,
            freeBlock.Text,
            "어둠 속에서 길을 잃었군.",
            "caption",
            true);

    Check("Translation 0023 rejects placeholder output",
        !TranslationRefinementService.IsUsableTranslation(
            freeTranslation,
            "...[여기에 번역된 내용이 들어갑니다]"));

    var cantBlock =
        new OcrTextBlock(
            2006,
            74,
            160,
            205,
            30,
            "I CAN'T.",
            1,
            "en",
            [new OcrLine(
                74,
                160,
                205,
                30,
                "I CAN'T.",
                0.99f,
                "en")]);

    var cantTranslation =
        new VisionTranslation(
            2006,
            cantBlock,
            "I CAN'T.",
            "난 못 해.",
            "dialogue",
            true);

    Check("Translation 0023 rejects lost English negation",
        !TranslationRefinementService.IsUsableTranslation(
            cantTranslation,
            "난 할 수 있어."));

    Check("Translation 0023 accepts preserved English negation",
        TranslationRefinementService.IsUsableTranslation(
            cantTranslation,
            "난 할 수 없어."));

    var freeSelection =
        V2EraseSelector.Select(
            freeSnapshot,
            [freeTranslation]);

    Check("V2 selector binds approved free text without inventing a Bubble",
        freeSelection.Bindings.TryGetValue(
            2002,
            out var freeBinding) &&
        freeBinding.TextRegionIds.SequenceEqual(["VT-FREE"]) &&
        freeBinding.BubbleRegionId is null &&
        freeBinding.LayoutMode == "rtdetr_textfree" &&
        freeBinding.LayoutBounds == freeTextRegion.Bounds);

    var stylizedTranslation =
        new VisionTranslation(
            2003,
            freeBlock,
            "BRUCE!",
            "브루스!",
            "dialogue",
            true);

    var stylizedSelection =
        V2EraseSelector.Select(
            freeSnapshot,
            [stylizedTranslation]);

    Check("V2 selector preserves free-standing stylized shout text",
        !stylizedSelection.Bindings.ContainsKey(2003) &&
        stylizedSelection.PreservationReasons.TryGetValue(
            2003,
            out var stylizedReason) &&
        stylizedReason == "stylized_graphic");

    var ordinaryUppercaseTranslation =
        new VisionTranslation(
            2005,
            freeBlock,
            "AND DEATH WILL BE UPON YOU!",
            "죽음이 닥칠 것이다!",
            "dialogue",
            true);

    var ordinaryUppercaseSelection =
        V2EraseSelector.Select(
            freeSnapshot,
            [ordinaryUppercaseTranslation]);

    Check("V2 selector does not misclassify long uppercase TextFree dialogue as graphic",
        ordinaryUppercaseSelection.Bindings.ContainsKey(2005) &&
        !ordinaryUppercaseSelection.PreservationReasons.ContainsKey(2005));

    var shortCaptionTranslation =
        new VisionTranslation(
            2004,
            freeBlock,
            "THAT'S ALL.",
            "그게 전부야.",
            "caption",
            true);

    var shortCaptionSelection =
        V2EraseSelector.Select(
            freeSnapshot,
            [shortCaptionTranslation]);

    Check("V2 selector still translates short TextFree captions",
        shortCaptionSelection.Bindings.ContainsKey(2004) &&
        !shortCaptionSelection.PreservationReasons.ContainsKey(2004));

    var v2MainResult =
        new ErasePipelineV2().Run(
            v2MainSourcePath,
            v2MainRoot,
            mainSnapshot,
            selection.TextRegionIds);

    Check("V2 main erase writes first residual and final debug stages",
        File.Exists(v2MainResult.MaskDebugPath) &&
        File.Exists(v2MainResult.FirstCleanedDebugPath) &&
        File.Exists(v2MainResult.ResidualDebugPath) &&
        File.Exists(v2MainResult.CleanedDebugPath) &&
        File.Exists(v2MainResult.DetectionJsonPath));

    Check("V2 main erase reports only selected target",
        v2MainResult.DetectedTargetCount == 2 &&
        v2MainResult.TargetCount == 1 &&
        v2MainResult.MaskPixels > 0 &&
        v2MainResult.TargetAudits.Count == 1 &&
        v2MainResult.TargetAudits[0].InitialMaskPixels > 0);

    using var v2Final =
        Cv2.ImRead(
            v2MainResult.CleanedDebugPath,
            ImreadModes.Color);

    Check("V2 main erase preserves unselected TextBubble pixels",
        !v2Final.Empty() &&
        v2Final.At<Vec3b>(104, 258).Equals(
            v2MainSource.At<Vec3b>(104, 258)));
}
finally
{
    try
    {
        if (Directory.Exists(v2MainRoot))
            Directory.Delete(v2MainRoot, true);
    }
    catch
    {
    }
}

string flatTextFreeRoot = Path.Combine(
    Path.GetTempPath(),
    $"lmt_v2_flat_textfree_{Guid.NewGuid():N}");
try
{
    OutputDirectoryLayout.Ensure(
        flatTextFreeRoot);

    string flatSourcePath =
        Path.Combine(
            flatTextFreeRoot,
            "flat_textfree.png");

    using var flatSource =
        Mat.Zeros(
            220,
            480,
            MatType.CV_8UC3)
        .ToMat();

    flatSource.SetTo(
        new Scalar(
            12,
            12,
            12));

    Cv2.PutText(
        flatSource,
        "THAT PETITE BODY WILL CRUMBLE",
        new Point(
            60,
            120),
        HersheyFonts.HersheySimplex,
        0.65,
        new Scalar(
            245,
            245,
            245),
        2,
        LineTypes.AntiAlias);

    Cv2.ImWrite(
        flatSourcePath,
        flatSource);

    var flatTextRegion =
        new PageRegion(
            "VT-FLAT-DARK",
            PageRegionKind.TextFree,
            new Rect(
                50,
                82,
                365,
                55),
            0.95f,
            "test-rtdetr");

    var flatSnapshot =
        V2DetectionSnapshot.Create(
            new PageAnalysisResult(
                [flatTextRegion],
                [],
                "v2-flat-dark-test",
                true));

    var flatResult =
        new ErasePipelineV2().Run(
            flatSourcePath,
            flatTextFreeRoot,
            flatSnapshot,
            new HashSet<string>(
                ["VT-FLAT-DARK"],
                StringComparer.Ordinal));

    Check("V2 0022 flat TextFree dark panel erases cleanly",
        flatResult.TargetAudits.Count == 1 &&
        ErasePipelineV2.IsAuditClean(
            flatResult.TargetAudits[0]));
}
finally
{
    try
    {
        if (Directory.Exists(
                flatTextFreeRoot))
        {
            Directory.Delete(
                flatTextFreeRoot,
                true);
        }
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

    string auditCommitted =
        Path.Combine(
            OutputDirectoryLayout.Debug(auditRoot),
            "audit_source.v2_05_committed_cleaned.webp");

    using (var committed = Cv2.ImRead(
               auditSource,
               ImreadModes.Color))
    {
        Cv2.Rectangle(
            committed,
            new Rect(10, 10, 10, 8),
            new Scalar(255, 255, 255),
            -1);

        Cv2.ImWrite(
            auditCommitted,
            committed,
            [new ImageEncodingParam(
                ImwriteFlags.WebPQuality,
                101)]);
    }

    await new FinalAuditService().GenerateAsync(
        auditSource,
        auditFinal,
        auditRoot,
        emptyStage,
        []);

    string auditCompare =
        Path.Combine(
            OutputDirectoryLayout.Audit(auditRoot),
            "audit_source.final_compare.webp");

    string auditJson =
        Path.Combine(
            OutputDirectoryLayout.Audit(auditRoot),
            "audit_source.final_audit.json");

    using var auditCompareImage =
        Cv2.ImDecode(
            File.ReadAllBytes(
                auditCompare),
            ImreadModes.Color);

    string auditJsonText =
        File.ReadAllText(
            auditJson);

    Check("final audit writes four-stage original erase final diff sheet",
        File.Exists(auditCompare) &&
        !auditCompareImage.Empty() &&
        auditCompareImage.Cols == 200 &&
        auditJsonText.Contains(
            "\"schema\": \"final-audit-v2\"",
            StringComparison.Ordinal) &&
        auditJsonText.Contains(
            "\"committed_cleaned_available\": true",
            StringComparison.Ordinal) &&
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
        Environment.GetEnvironmentVariable("LMT_BABERU_SMOKE"),
        "1",
        StringComparison.Ordinal))
{
    string? root = Environment.GetEnvironmentVariable("LMT_BABERU_MODEL_ROOT");
    if (string.IsNullOrWhiteSpace(root))
        throw new Exception("LMT_BABERU_MODEL_ROOT missing for Baberu smoke");

    await ExternalModelManager.EnsureBaberuAsync();

    string sample = Path.Combine(
        Path.GetTempPath(),
        $"lmt_baberu_smoke_{Guid.NewGuid():N}.png");

    try
    {
        using var smoke = Mat.Zeros(240, 320, MatType.CV_8UC3).ToMat();
        smoke.SetTo(new Scalar(255, 255, 255));
        Cv2.PutText(
            smoke,
            "HELLO",
            new Point(82, 132),
            HersheyFonts.HersheySimplex,
            1.2,
            new Scalar(0, 0, 0),
            2,
            LineTypes.AntiAlias);
        Cv2.ImWrite(sample, smoke);

        var bubble = new PageRegion(
            "RG900",
            PageRegionKind.Bubble,
            new Rect(35, 50, 250, 130),
            0.95f,
            "smoke");

        var textBubble = new PageRegion(
            "RG901",
            PageRegionKind.TextBubble,
            new Rect(65, 92, 150, 52),
            0.95f,
            "smoke");

        using var baberu = new BaberuOcrEngine();

        var result = baberu.Analyze(
            sample,
            new PageAnalysisResult(
                [bubble, textBubble],
                [],
                "smoke",
                true));

        Check("Baberu ONNX sessions and decode loop smoke",
            result.Count == 1 &&
            result[0].RegionId == "RG900");
    }
    finally
    {
        try
        {
            if (File.Exists(sample))
                File.Delete(sample);
        }
        catch
        {
        }
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
