using LocalMangaTranslator.Models;

namespace LocalMangaTranslator.Services;

public sealed class OcrBlockGrouper
{
    public List<OcrTextBlock> Group(IReadOnlyList<OcrLine> input)
    {
        if (input.Count == 0)
            return [];

        // 예전처럼 "붙은 그룹에 다시 붙이고 또 붙이는" transitive merge를 하지 않는다.
        // 각 OCR line은 가장 적합한 단 하나의 기존 그룹에만 들어갈 수 있다.
        var ordered = input
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .ToList();

        var groups = new List<List<OcrLine>>();

        foreach (var line in ordered)
        {
            int bestIndex = -1;
            double bestScore = double.NegativeInfinity;

            for (int i = 0; i < groups.Count; i++)
            {
                double score = JoinScore(groups[i], line);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = i;
                }
            }

            if (bestIndex >= 0 && bestScore >= 1.0)
                groups[bestIndex].Add(line);
            else
                groups.Add([line]);
        }

        // 마지막으로 한 그룹이 비정상적으로 퍼졌는지 확인하고,
        // 그렇다면 강한 연결만 남겨 다시 분리한다.
        var refined = new List<List<OcrLine>>();

        foreach (var group in groups)
            refined.AddRange(SplitSuspiciousGroup(group));

        return refined
            .Select(OrderLines)
            .Select((lines, id) => ToBlock(id, lines))
            .OrderBy(x => x.Y)
            .ThenBy(x => x.X)
            .Select((block, id) => block with { Id = id })
            .ToList();
    }

    static double JoinScore(
        IReadOnlyList<OcrLine> group,
        OcrLine candidate)
    {
        if (group.Count == 0)
            return double.NegativeInfinity;

        var bounds = Bounds(group);

        double medianHeight = Median(group.Select(x => x.H));
        double medianWidth = Median(group.Select(x => x.W));

        bool groupVertical =
            group.Count(x => IsVertical(x)) >
            group.Count / 2;

        bool candidateVertical = IsVertical(candidate);

        // 거대한 효과음과 작은 대사를 섞지 않는다.
        double sizeBasisGroup =
            groupVertical
                ? Math.Max(4, medianWidth)
                : Math.Max(4, medianHeight);

        double sizeBasisCandidate =
            candidateVertical
                ? Math.Max(4, candidate.W)
                : Math.Max(4, candidate.H);

        double sizeRatio =
            Math.Max(sizeBasisGroup, sizeBasisCandidate) /
            Math.Max(1, Math.Min(sizeBasisGroup, sizeBasisCandidate));

        if (sizeRatio > 2.15)
            return double.NegativeInfinity;

        if (group.Count >= 2 &&
            groupVertical != candidateVertical &&
            (IsStronglyVertical(candidate) ||
             group.Any(IsStronglyVertical)))
            return double.NegativeInfinity;

        double candidateLeft = candidate.X;
        double candidateRight = candidate.X + candidate.W;
        double candidateTop = candidate.Y;
        double candidateBottom = candidate.Y + candidate.H;

        double horizontalOverlap = OverlapRatio(
            bounds.X,
            bounds.Right,
            candidateLeft,
            candidateRight,
            Math.Min(bounds.Right - bounds.X, candidate.W));

        double verticalOverlap = OverlapRatio(
            bounds.Y,
            bounds.Bottom,
            candidateTop,
            candidateBottom,
            Math.Min(bounds.Bottom - bounds.Y, candidate.H));

        double verticalGap = Gap(
            bounds.Y,
            bounds.Bottom,
            candidateTop,
            candidateBottom);

        double horizontalGap = Gap(
            bounds.X,
            bounds.Right,
            candidateLeft,
            candidateRight);

        double groupCenterX =
            (bounds.X + bounds.Right) / 2.0;

        double groupCenterY =
            (bounds.Y + bounds.Bottom) / 2.0;

        double candidateCenterX =
            candidate.X + candidate.W / 2.0;

        double candidateCenterY =
            candidate.Y + candidate.H / 2.0;

        double centerDeltaX =
            Math.Abs(groupCenterX - candidateCenterX);

        double centerDeltaY =
            Math.Abs(groupCenterY - candidateCenterY);

        double score;

        if (groupVertical && candidateVertical)
        {
            // 세로 일본어는 인접한 세로 열끼리만 묶는다.
            bool adjacentColumns =
                horizontalGap <= Math.Max(10, medianWidth * 0.85) &&
                (verticalOverlap >= 0.42 ||
                 centerDeltaY <= Math.Max(bounds.Bottom - bounds.Y, candidate.H) * 0.30);

            bool sameColumnFragment =
                horizontalOverlap >= 0.62 &&
                verticalGap <= Math.Max(12, medianWidth * 0.90);

            if (!adjacentColumns && !sameColumnFragment)
                return double.NegativeInfinity;

            score =
                verticalOverlap * 2.1 +
                horizontalOverlap * 1.2 -
                horizontalGap / Math.Max(10, medianWidth) * 0.75 -
                verticalGap / Math.Max(12, medianWidth) * 0.35;
        }
        else
        {
            // 일반 가로 대사는 줄 간격과 중심 정렬을 꽤 엄격하게 본다.
            bool stackedLine =
                verticalGap <= Math.Max(10, medianHeight * 0.58) &&
                (horizontalOverlap >= 0.38 ||
                 centerDeltaX <= Math.Max(medianWidth, candidate.W) * 0.30);

            // 한 줄이 OCR에서 둘로 쪼개진 경우. 서로 다른 말풍선이 같은 높이에
            // 있다고 붙지 않도록 기존보다 수평 gap 조건을 크게 줄였다.
            bool sameRowFragment =
                verticalOverlap >= 0.70 &&
                horizontalGap <= Math.Max(12, medianHeight * 0.72);

            if (!stackedLine && !sameRowFragment)
                return double.NegativeInfinity;

            score =
                horizontalOverlap * 2.0 +
                verticalOverlap * 1.25 -
                verticalGap / Math.Max(10, medianHeight) * 0.90 -
                horizontalGap / Math.Max(12, medianHeight) * 0.55;
        }

        var tentative = group
            .Append(candidate)
            .ToList();

        var t = Bounds(tentative);

        double sumArea = tentative.Sum(
            x => Math.Max(1.0, x.W * x.H));

        double bboxArea = Math.Max(
            1.0,
            (t.Right - t.X) * (t.Bottom - t.Y));

        double density = sumArea / bboxArea;

        // 서로 다른 말풍선 여러 개를 한 박스로 감싸면 density가 급격히 낮아진다.
        if (tentative.Count >= 3 && density < 0.16)
            return double.NegativeInfinity;

        if (!groupVertical)
        {
            double maxHeight =
                Math.Max(160, medianHeight * 7.0);

            double maxWidth =
                Math.Max(260, tentative.Max(x => x.W) * 2.30);

            if (t.Bottom - t.Y > maxHeight ||
                t.Right - t.X > maxWidth)
                return double.NegativeInfinity;
        }
        else
        {
            double maxWidth =
                Math.Max(160, medianWidth * 7.0);

            double maxHeight =
                Math.Max(260, tentative.Max(x => x.H) * 2.40);

            if (t.Right - t.X > maxWidth ||
                t.Bottom - t.Y > maxHeight)
                return double.NegativeInfinity;
        }

        score += Math.Clamp(density, 0, 1) * 0.8;
        score -= Math.Max(0, sizeRatio - 1.0) * 0.45;

        return score;
    }

    static IEnumerable<List<OcrLine>> SplitSuspiciousGroup(
        IReadOnlyList<OcrLine> group)
    {
        if (group.Count <= 1)
        {
            yield return group.ToList();
            yield break;
        }

        var b = Bounds(group);
        double medianHeight = Median(group.Select(x => x.H));
        double medianWidth = Median(group.Select(x => x.W));
        bool vertical = group.Count(IsVertical) > group.Count / 2;

        double sumArea = group.Sum(
            x => Math.Max(1.0, x.W * x.H));

        double bboxArea = Math.Max(
            1.0,
            (b.Right - b.X) * (b.Bottom - b.Y));

        double density = sumArea / bboxArea;

        bool suspicious = vertical
            ? b.Right - b.X > Math.Max(150, medianWidth * 5.5) ||
              density < 0.14
            : b.Bottom - b.Y > Math.Max(135, medianHeight * 6.0) ||
              density < 0.14;

        if (!suspicious)
        {
            yield return group.ToList();
            yield break;
        }

        // 강한 pair 연결만으로 connected component를 다시 만든다.
        var components = new List<List<OcrLine>>();

        foreach (var line in group.OrderBy(x => x.Y).ThenBy(x => x.X))
        {
            int best = -1;
            double bestScore = double.NegativeInfinity;

            for (int i = 0; i < components.Count; i++)
            {
                double score = JoinScore(components[i], line);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = i;
                }
            }

            if (best >= 0 && bestScore >= 1.35)
                components[best].Add(line);
            else
                components.Add([line]);
        }

        foreach (var component in components)
            yield return component;
    }

    static bool IsVertical(OcrLine line)
        => line.H > line.W * 1.30;

    static bool IsStronglyVertical(OcrLine line)
        => line.H > line.W * 1.75;

    static List<OcrLine> OrderLines(
        IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();

        bool vertical =
            list.Count >= 2 &&
            list.Count(IsVertical) > list.Count / 2;

        return vertical
            ? list
                .OrderByDescending(x => x.X)
                .ThenBy(x => x.Y)
                .ToList()
            : list
                .OrderBy(x => x.Y)
                .ThenBy(x => x.X)
                .ToList();
    }

    static OcrTextBlock ToBlock(
        int id,
        IReadOnlyList<OcrLine> lines)
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

    static string JoinText(
        IReadOnlyList<OcrLine> lines)
    {
        return string.Join(
                " ",
                lines
                    .Select(x => x.Text.Trim())
                    .Where(x => !string.IsNullOrWhiteSpace(x)))
            .Replace("  ", " ")
            .Trim();
    }

    static (
        double X,
        double Y,
        double Right,
        double Bottom) Bounds(
        IEnumerable<OcrLine> lines)
    {
        var list = lines.ToList();

        return (
            list.Min(x => x.X),
            list.Min(x => x.Y),
            list.Max(x => x.X + x.W),
            list.Max(x => x.Y + x.H));
    }

    static double Gap(
        double a1,
        double a2,
        double b1,
        double b2)
    {
        if (a2 < b1)
            return b1 - a2;

        if (b2 < a1)
            return a1 - b2;

        return 0;
    }

    static double OverlapRatio(
        double a1,
        double a2,
        double b1,
        double b2,
        double basis)
    {
        if (basis <= 0)
            return 0;

        double overlap = Math.Max(
            0,
            Math.Min(a2, b2) - Math.Max(a1, b1));

        return overlap / basis;
    }

    static double Median(
        IEnumerable<double> source)
    {
        var values = source
            .OrderBy(x => x)
            .ToArray();

        if (values.Length == 0)
            return 0;

        int mid = values.Length / 2;

        return values.Length % 2 == 1
            ? values[mid]
            : (values[mid - 1] + values[mid]) / 2.0;
    }
}
