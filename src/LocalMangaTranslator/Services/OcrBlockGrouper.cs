using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class OcrBlockGrouper
{
    public List<OcrTextBlock> Group(IReadOnlyList<OcrLine> input)
    {
        if (input.Count == 0) return [];

        var remaining = input
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .ToList();

        var groups = new List<List<OcrLine>>();

        while (remaining.Count > 0)
        {
            var group = new List<OcrLine> { remaining[0] };
            remaining.RemoveAt(0);

            bool changed;
            do
            {
                changed = false;

                for (int i = remaining.Count - 1; i >= 0; i--)
                {
                    if (!ShouldJoin(group, remaining[i])) continue;
                    group.Add(remaining[i]);
                    remaining.RemoveAt(i);
                    changed = true;
                }
            }
            while (changed);

            groups.Add(OrderLines(group));
        }

        return groups
            .Select((lines, id) => ToBlock(id, lines))
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .Select((block, id) => block with { Id = id })
            .ToList();
    }

    static bool ShouldJoin(IReadOnlyList<OcrLine> group, OcrLine candidate)
    {
        var bounds = Bounds(group);
        double avgHeight = Math.Max(8, group.Average(x => x.H));
        double avgWidth = Math.Max(8, group.Average(x => x.W));

        double candidateLeft = candidate.X;
        double candidateRight = candidate.X + candidate.W;
        double candidateTop = candidate.Y;
        double candidateBottom = candidate.Y + candidate.H;

        double horizontalOverlap = OverlapRatio(
            bounds.X, bounds.Right,
            candidateLeft, candidateRight,
            Math.Min(bounds.Right - bounds.X, candidate.W));

        double verticalOverlap = OverlapRatio(
            bounds.Y, bounds.Bottom,
            candidateTop, candidateBottom,
            Math.Min(bounds.Bottom - bounds.Y, candidate.H));

        double verticalGap = Gap(bounds.Y, bounds.Bottom, candidateTop, candidateBottom);
        double horizontalGap = Gap(bounds.X, bounds.Right, candidateLeft, candidateRight);

        double groupCenterX = (bounds.X + bounds.Right) / 2;
        double candidateCenterX = candidate.X + candidate.W / 2;
        double centerDeltaX = Math.Abs(groupCenterX - candidateCenterX);

        bool stackedLine =
            verticalGap <= Math.Max(18, avgHeight * 0.72) &&
            (horizontalOverlap >= 0.24 ||
             centerDeltaX <= Math.Max(avgWidth, candidate.W) * 0.42);

        bool sameRowFragment =
            verticalOverlap >= 0.55 &&
            horizontalGap <= Math.Max(28, avgHeight * 1.15);

        if (!stackedLine && !sameRowFragment)
            return false;

        var tentative = group.Append(candidate).ToList();
        var t = Bounds(tentative);

        double medianHeight = tentative.Select(x => x.H).OrderBy(x => x).ElementAt(tentative.Count / 2);

        if (t.Bottom - t.Y > Math.Max(520, medianHeight * 9.0))
            return false;

        if (t.Right - t.X > Math.Max(900, tentative.Max(x => x.W) * 3.2))
            return false;

        return true;
    }

    static List<OcrLine> OrderLines(IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();
        bool vertical = list.Count >= 2 &&
                        list.Average(x => x.H) > list.Average(x => x.W) * 1.25;

        return vertical
            ? list.OrderByDescending(x => x.X).ThenBy(x => x.Y).ToList()
            : list.OrderBy(x => x.Y).ThenBy(x => x.X).ToList();
    }

    static OcrTextBlock ToBlock(int id, IReadOnlyList<OcrLine> lines)
    {
        var b = Bounds(lines);
        string text = JoinText(lines);
        string language = lines
            .GroupBy(x => x.Language)
            .OrderByDescending(x => x.Count())
            .Select(x => x.Key)
            .FirstOrDefault() ?? "en";

        return new OcrTextBlock(
            id,
            b.X,
            b.Y,
            Math.Max(1, b.Right - b.X),
            Math.Max(1, b.Bottom - b.Y),
            text,
            lines.Count,
            language,
            lines.ToList());
    }

    static string JoinText(IReadOnlyList<OcrLine> lines)
    {
        return string.Join(" ", lines
            .Select(x => x.Text.Trim())
            .Where(x => !string.IsNullOrWhiteSpace(x)))
            .Replace("  ", " ")
            .Trim();
    }

    static (double X, double Y, double Right, double Bottom) Bounds(IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();
        return (
            list.Min(x => x.X),
            list.Min(x => x.Y),
            list.Max(x => x.X + x.W),
            list.Max(x => x.Y + x.H));
    }

    static double Gap(double a1, double a2, double b1, double b2)
    {
        if (a2 < b1) return b1 - a2;
        if (b2 < a1) return a1 - b2;
        return 0;
    }

    static double OverlapRatio(double a1, double a2, double b1, double b2, double basis)
    {
        if (basis <= 0) return 0;
        double overlap = Math.Max(0, Math.Min(a2, b2) - Math.Max(a1, b1));
        return overlap / basis;
    }
}
