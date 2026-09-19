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
using var cancellation = new CancellationTokenSource();
cancellation.Cancel();
bool cancelled = false;
try { ContainerOcrValidator.ValidateLines([candidate], [a, b], cancellation.Token); }
catch (OperationCanceledException) { cancelled = true; }
Check("cancellation propagates", cancelled);
Console.WriteLine($"{count} pipeline checks passed.");
