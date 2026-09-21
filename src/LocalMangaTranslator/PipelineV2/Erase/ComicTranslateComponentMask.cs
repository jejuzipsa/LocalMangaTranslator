using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

public enum ComicTranslateMaskLayer
{
    Merged,
    Grayscale,
    ColorRescue
}

/// <summary>
/// Text-pixel mask inside an immutable RT-DETR TextBubble.
///
/// Important safety rules:
/// - choose ONE grayscale foreground polarity, never OR black/white masks together;
/// - optionally add a tightly filtered chroma-contrast rescue for colored glyphs;
/// - reject masks that consume most of the detector text box;
/// - keep all pixels clamped to immutable detector geometry.
/// </summary>
public static class ComicTranslateComponentMask
{
    public static Mat Build(
        Mat source,
        Rect textBounds,
        Rect? bubbleBounds = null,
        bool includeColorRescue = true,
        bool dilateMask = true,
        ComicTranslateMaskLayer layer =
            ComicTranslateMaskLayer.Merged)
    {
        var cropBounds =
            ClampRect(
                textBounds.X - 10,
                textBounds.Y - 10,
                textBounds.Width + 20,
                textBounds.Height + 20,
                source.Cols,
                source.Rows);

        if (bubbleBounds.HasValue)
        {
            var clipped =
                Intersect(
                    cropBounds,
                    bubbleBounds.Value);

            if (clipped.Width > 0 &&
                clipped.Height > 0)
            {
                cropBounds = clipped;
            }
        }

        var full =
            Mat.Zeros(
                    source.Rows,
                    source.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        if (cropBounds.Width <= 0 ||
            cropBounds.Height <= 0)
        {
            return full;
        }

        using var crop =
            new Mat(
                source,
                cropBounds);

        using var gray =
            new Mat();

        Cv2.CvtColor(
            crop,
            gray,
            ColorConversionCodes.BGR2GRAY);

        var textLocal =
            new Rect(
                Math.Max(
                    0,
                    textBounds.X -
                    cropBounds.X),
                Math.Max(
                    0,
                    textBounds.Y -
                    cropBounds.Y),
                Math.Min(
                    textBounds.Right,
                    cropBounds.Right) -
                Math.Max(
                    textBounds.Left,
                    cropBounds.Left),
                Math.Min(
                    textBounds.Bottom,
                    cropBounds.Bottom) -
                Math.Max(
                    textBounds.Top,
                    cropBounds.Top));

        if (textLocal.Width <= 0 ||
            textLocal.Height <= 0)
        {
            return full;
        }

        using var blackBinary =
            new Mat();

        using var whiteBinary =
            new Mat();

        Cv2.Threshold(
            gray,
            blackBinary,
            0,
            255,
            ThresholdTypes.BinaryInv |
            ThresholdTypes.Otsu);

        Cv2.Threshold(
            gray,
            whiteBinary,
            0,
            255,
            ThresholdTypes.Binary |
            ThresholdTypes.Otsu);

        using var blackCandidate =
            BuildComponentCandidate(
                blackBinary,
                textLocal);

        using var whiteCandidate =
            BuildComponentCandidate(
                whiteBinary,
                textLocal);

        double background =
            EstimateBackgroundMedian(
                gray,
                textLocal);

        using var chosen =
            ChoosePolarity(
                blackCandidate,
                whiteCandidate,
                textLocal,
                background);

        if (Cv2.CountNonZero(chosen) < 2)
        {
            AddPolarityContrastFallback(
                gray,
                chosen,
                textLocal,
                background);
        }

        // Colored comic lettering can have a mid luminance and therefore
        // escape both black/white Otsu masks. Keep the actual foreground
        // pixels instead of filling their external contour; comic lettering
        // such as O/P/R contains meaningful white holes that must never become
        // erase pixels. A small dark-outline rescue grows only from accepted
        // colored seed pixels, which captures black ink around red lettering
        // without flood-filling the white negative space.
        //
        // Residual review still calls Build(..., includeColorRescue: false)
        // and only asks whether normal high-contrast glyph structure remains.
        using var colorCandidate =
            Mat.Zeros(
                    crop.Rows,
                    crop.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        if (includeColorRescue)
        {
            using var detectedColor =
                BuildColorContrastCandidate(
                    crop,
                    gray,
                    textLocal,
                    background,
                    conservative:
                        !bubbleBounds.HasValue);

            int colorPixels =
                Cv2.CountNonZero(
                    detectedColor);

            int preColorArea =
                Math.Max(
                    1,
                    textLocal.Width *
                    textLocal.Height);

            if (colorPixels >= 2 &&
                colorPixels <=
                    preColorArea *
                    (bubbleBounds.HasValue
                        ? 0.30
                        : 0.20))
            {
                detectedColor.CopyTo(
                    colorCandidate);
            }
        }

        using var selected =
            layer switch
            {
                ComicTranslateMaskLayer.Grayscale =>
                    chosen.Clone(),
                ComicTranslateMaskLayer.ColorRescue =>
                    colorCandidate.Clone(),
                _ =>
                    chosen.Clone()
            };

        if (layer ==
            ComicTranslateMaskLayer.Merged)
        {
            using var merged =
                new Mat();

            Cv2.BitwiseOr(
                selected,
                colorCandidate,
                merged);

            int mergedPixels =
                Cv2.CountNonZero(
                    merged);

            int preColorArea =
                Math.Max(
                    1,
                    textLocal.Width *
                    textLocal.Height);

            if (mergedPixels <=
                preColorArea * 0.45)
            {
                merged.CopyTo(
                    selected);
            }
        }

        // Ambiguous segmentation must never erase most of the detector box.
        int textArea =
            Math.Max(
                1,
                textLocal.Width *
                textLocal.Height);

        int chosenPixels =
            Cv2.CountNonZero(
                selected);

        if (chosenPixels >
            textArea * 0.48)
        {
            selected.SetTo(
                Scalar.Black);
        }

        if (dilateMask)
        {
            using var kernel =
                Cv2.GetStructuringElement(
                    MorphShapes.Ellipse,
                    new Size(3, 3));

            Cv2.Dilate(
                selected,
                selected,
                kernel,
                iterations: 1);
        }

        // Erase dilation can grow outside TextBubble, so clamp once more.
        // Residual review asks for dilateMask:false to keep an undilated
        // glyph core that represents where source lettering actually lived.
        using var clamped =
            Mat.Zeros(
                    gray.Rows,
                    gray.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        using (var srcRoi =
               new Mat(
                   selected,
                   textLocal))
        using (var dstRoi =
               new Mat(
                   clamped,
                   textLocal))
        {
            srcRoi.CopyTo(
                dstRoi);
        }

        using var destination =
            new Mat(
                full,
                cropBounds);

        clamped.CopyTo(
            destination);

        return full;
    }

    static Mat BuildComponentCandidate(
        Mat binary,
        Rect textLocal)
    {
        var candidate =
            Mat.Zeros(
                    binary.Rows,
                    binary.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        Cv2.FindContours(
            binary,
            out Point[][] contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        double cropArea =
            Math.Max(
                1.0,
                binary.Rows *
                (double)binary.Cols);

        foreach (var contour in contours)
        {
            if (contour.Length == 0)
                continue;

            var bounds =
                Cv2.BoundingRect(
                    contour);

            double overlap =
                IntersectionArea(
                    bounds,
                    textLocal);

            if (overlap <= 0)
                continue;

            double boundsArea =
                Math.Max(
                    1.0,
                    bounds.Width *
                    (double)bounds.Height);

            if (overlap /
                boundsArea <
                0.35)
            {
                continue;
            }

            using var componentRoi =
                new Mat(
                    binary,
                    bounds);

            int pixelArea =
                Cv2.CountNonZero(
                    componentRoi);

            bool ordinary =
                pixelArea > 10;

            bool punctuation =
                pixelArea >= 2 &&
                bounds.Width <= 8 &&
                bounds.Height <= 8;

            if (!ordinary &&
                !punctuation)
            {
                continue;
            }

            const int margin = 1;

            bool touchesCropBorder =
                bounds.Left < margin ||
                bounds.Top < margin ||
                bounds.Right >
                    binary.Cols - margin ||
                bounds.Bottom >
                    binary.Rows - margin;

            if (touchesCropBorder)
                continue;

            // Large connected backgrounds are not text.
            if (pixelArea >=
                cropArea * 0.35)
            {
                continue;
            }

            // 0029: select this connected shape, but copy only the
            // foreground pixels that were actually present in the binary
            // source. Filling an external contour turned O/P/R holes and
            // narrow inter-letter gaps into erase pixels.
            using var componentShape =
                Mat.Zeros(
                        binary.Rows,
                        binary.Cols,
                        MatType.CV_8UC1)
                    .ToMat();

            Cv2.DrawContours(
                componentShape,
                [contour],
                -1,
                Scalar.White,
                thickness: -1);

            Cv2.BitwiseAnd(
                componentShape,
                binary,
                componentShape);

            Cv2.BitwiseOr(
                candidate,
                componentShape,
                candidate);
        }

        using var restricted =
            Mat.Zeros(
                    binary.Rows,
                    binary.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        using (var srcRoi =
               new Mat(
                   candidate,
                   textLocal))
        using (var dstRoi =
               new Mat(
                   restricted,
                   textLocal))
        {
            srcRoi.CopyTo(
                dstRoi);
        }

        candidate.Dispose();

        return restricted.Clone();
    }

    static Mat ChoosePolarity(
        Mat blackCandidate,
        Mat whiteCandidate,
        Rect textLocal,
        double background)
    {
        int textArea =
            Math.Max(
                1,
                textLocal.Width *
                textLocal.Height);

        int blackPixels =
            Cv2.CountNonZero(
                blackCandidate);

        int whitePixels =
            Cv2.CountNonZero(
                whiteCandidate);

        bool blackValid =
            blackPixels >= 2 &&
            blackPixels <=
                textArea * 0.45;

        bool whiteValid =
            whitePixels >= 2 &&
            whitePixels <=
                textArea * 0.45;

        Mat? selected = null;

        if (background >= 145)
        {
            if (blackValid)
                selected = blackCandidate;
            else if (whiteValid)
                selected = whiteCandidate;
        }
        else if (background <= 110)
        {
            if (whiteValid)
                selected = whiteCandidate;
            else if (blackValid)
                selected = blackCandidate;
        }
        else
        {
            if (blackValid &&
                whiteValid)
            {
                selected =
                    blackPixels <=
                    whitePixels
                        ? blackCandidate
                        : whiteCandidate;
            }
            else if (blackValid)
            {
                selected =
                    blackCandidate;
            }
            else if (whiteValid)
            {
                selected =
                    whiteCandidate;
            }
        }

        return selected is null
            ? Mat.Zeros(
                    blackCandidate.Rows,
                    blackCandidate.Cols,
                    MatType.CV_8UC1)
                .ToMat()
            : selected.Clone();
    }

    static void AddPolarityContrastFallback(
        Mat gray,
        Mat destination,
        Rect textLocal,
        double background)
    {
        using var raw =
            Mat.Zeros(
                    gray.Rows,
                    gray.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        double threshold =
            background <= 110 ||
            background >= 145
                ? 30
                : 42;

        for (int y =
                 textLocal.Top;
             y <
             textLocal.Bottom;
             y++)
        {
            for (int x =
                     textLocal.Left;
                 x <
                 textLocal.Right;
                 x++)
            {
                double value =
                    gray.At<byte>(
                        y,
                        x);

                bool likelyText =
                    background <= 110
                        ? value >=
                          background +
                          threshold
                        : background >= 145
                            ? value <=
                              background -
                              threshold
                            : Math.Abs(
                                  value -
                                  background) >=
                              threshold;

                if (likelyText)
                {
                    raw.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        using var filtered =
            BuildComponentCandidate(
                raw,
                textLocal);

        int textArea =
            Math.Max(
                1,
                textLocal.Width *
                textLocal.Height);

        int pixels =
            Cv2.CountNonZero(
                filtered);

        if (pixels < 2 ||
            pixels >
            textArea * 0.45)
        {
            return;
        }

        Cv2.BitwiseOr(
            destination,
            filtered,
            destination);
    }

    static Mat BuildColorContrastCandidate(
        Mat color,
        Mat gray,
        Rect textLocal,
        double grayBackground,
        bool conservative)
    {
        var raw =
            Mat.Zeros(
                    color.Rows,
                    color.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        var blue =
            new List<byte>();

        var green =
            new List<byte>();

        var red =
            new List<byte>();

        for (int y =
                 textLocal.Top;
             y <
             textLocal.Bottom;
             y += 2)
        {
            for (int x =
                     textLocal.Left;
                 x <
                 textLocal.Right;
                 x += 2)
            {
                var pixel =
                    color.At<Vec3b>(
                        y,
                        x);

                blue.Add(
                    pixel.Item0);

                green.Add(
                    pixel.Item1);

                red.Add(
                    pixel.Item2);
            }
        }

        if (blue.Count == 0)
            return raw;

        blue.Sort();
        green.Sort();
        red.Sort();

        double bgB =
            blue[
                blue.Count / 2];

        double bgG =
            green[
                green.Count / 2];

        double bgR =
            red[
                red.Count / 2];

        double distanceThreshold =
            conservative
                ? 105
                : 72;

        double chromaThreshold =
            conservative
                ? 52
                : 34;

        for (int y =
                 textLocal.Top;
             y <
             textLocal.Bottom;
             y++)
        {
            for (int x =
                     textLocal.Left;
                 x <
                 textLocal.Right;
                 x++)
            {
                var pixel =
                    color.At<Vec3b>(
                        y,
                        x);

                double b =
                    pixel.Item0;

                double g =
                    pixel.Item1;

                double r =
                    pixel.Item2;

                double db =
                    b - bgB;

                double dg =
                    g - bgG;

                double dr =
                    r - bgR;

                double distance =
                    Math.Sqrt(
                        db * db +
                        dg * dg +
                        dr * dr);

                double max =
                    Math.Max(
                        r,
                        Math.Max(
                            g,
                            b));

                double min =
                    Math.Min(
                        r,
                        Math.Min(
                            g,
                            b));

                double chroma =
                    max - min;

                if (distance >=
                        distanceThreshold &&
                    chroma >=
                        chromaThreshold)
                {
                    raw.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        using var filtered =
            BuildComponentCandidate(
                raw,
                textLocal);

        using var outlineCandidate =
            BuildAdjacentOutlineCandidate(
                gray,
                filtered,
                textLocal,
                grayBackground);

        using var combined =
            new Mat();

        Cv2.BitwiseOr(
            filtered,
            outlineCandidate,
            combined);

        raw.Dispose();

        return combined.Clone();
    }

    static Mat BuildAdjacentOutlineCandidate(
        Mat gray,
        Mat colorSeed,
        Rect textLocal,
        double background)
    {
        var contrast =
            Mat.Zeros(
                    gray.Rows,
                    gray.Cols,
                    MatType.CV_8UC1)
                .ToMat();

        double threshold =
            background >= 145
                ? 34
                : background <= 110
                    ? 34
                    : 42;

        for (int y =
                 textLocal.Top;
             y <
             textLocal.Bottom;
             y++)
        {
            for (int x =
                     textLocal.Left;
                 x <
                 textLocal.Right;
                 x++)
            {
                double value =
                    gray.At<byte>(
                        y,
                        x);

                bool outlineLike =
                    background >= 145
                        ? value <=
                          background -
                          threshold
                        : background <= 110
                            ? value >=
                              background +
                              threshold
                            : Math.Abs(
                                  value -
                                  background) >=
                              threshold;

                if (outlineLike)
                {
                    contrast.Set(
                        y,
                        x,
                        (byte)255);
                }
            }
        }

        int radius =
            Math.Clamp(
                (int)Math.Round(
                    Math.Min(
                        textLocal.Width,
                        textLocal.Height) /
                    60.0),
                1,
                3);

        using var neighborhood =
            new Mat();

        using (var kernel =
               Cv2.GetStructuringElement(
                   MorphShapes.Ellipse,
                   new Size(
                       radius * 2 + 1,
                       radius * 2 + 1)))
        {
            Cv2.Dilate(
                colorSeed,
                neighborhood,
                kernel,
                iterations: 1);
        }

        var outline =
            new Mat();

        Cv2.BitwiseAnd(
            contrast,
            neighborhood,
            outline);

        contrast.Dispose();

        return outline;
    }

    static double EstimateBackgroundMedian(
        Mat gray,
        Rect textLocal)
    {
        var samples =
            new List<byte>(
                Math.Max(
                    16,
                    textLocal.Width *
                    textLocal.Height /
                    4));

        for (int y =
                 textLocal.Top;
             y <
             textLocal.Bottom;
             y += 2)
        {
            for (int x =
                     textLocal.Left;
                 x <
                 textLocal.Right;
                 x += 2)
            {
                samples.Add(
                    gray.At<byte>(
                        y,
                        x));
            }
        }

        if (samples.Count == 0)
            return 127;

        samples.Sort();

        return samples[
            samples.Count /
            2];
    }

    static double IntersectionArea(
        Rect a,
        Rect b)
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
                   right -
                   left) *
               (double)Math.Max(
                   0,
                   bottom -
                   top);
    }

    static Rect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left =
            Math.Clamp(
                x,
                0,
                imageWidth);

        int top =
            Math.Clamp(
                y,
                0,
                imageHeight);

        int right =
            Math.Clamp(
                x +
                Math.Max(
                    0,
                    width),
                0,
                imageWidth);

        int bottom =
            Math.Clamp(
                y +
                Math.Max(
                    0,
                    height),
                0,
                imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(
                0,
                right -
                left),
            Math.Max(
                0,
                bottom -
                top));
    }

    static Rect Intersect(
        Rect a,
        Rect b)
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

        return new Rect(
            left,
            top,
            Math.Max(
                0,
                right -
                left),
            Math.Max(
                0,
                bottom -
                top));
    }
}
