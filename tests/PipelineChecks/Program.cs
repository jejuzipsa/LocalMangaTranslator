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
var rescuableLine = new OcrLine(173, 240, 30, 12, "BACK", 0.91f, "en");
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

using var cancellation = new CancellationTokenSource();
cancellation.Cancel();
bool cancelled = false;
try { ContainerOcrValidator.ValidateLines([candidate], [a, b], cancellation.Token); }
catch (OperationCanceledException) { cancelled = true; }
Check("cancellation propagates", cancelled);
Console.WriteLine($"{count} pipeline checks passed.");
