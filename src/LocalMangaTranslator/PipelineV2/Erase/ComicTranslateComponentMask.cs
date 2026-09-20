using OpenCvSharp;

namespace LocalMangaTranslator.PipelineV2.Erase;

/// <summary>
/// C# adaptation of Comic Translate's Apache-2.0 content-mask idea:
/// inside an already detected text bbox, use Otsu black/white masks and keep
/// text-sized connected components while rejecting border/background fills.
///
/// This code deliberately has no OCR, translation, ownership, or canonical-line
/// dependency. RT-DETR decides where text is; this class only decides which
/// pixels inside that detected area look like source lettering.
/// </summary>
public static class ComicTranslateComponentMask
{
    public static Mat Build(
        Mat source,
        Rect textBounds,
        Rect? bubbleBounds = null)
    {
        var cropBounds = ClampRect(
            textBounds.X - 10,
            textBounds.Y - 10,
            textBounds.Width + 20,
            textBounds.Height + 20,
            source.Cols,
            source.Rows);

        if (bubbleBounds.HasValue)
        {
            var clipped = Intersect(cropBounds, bubbleBounds.Value);
            if (clipped.Width > 0 && clipped.Height > 0)
                cropBounds = clipped;
        }

        var full = Mat.Zeros(
            source.Rows,
            source.Cols,
            MatType.CV_8UC1).ToMat();

        if (cropBounds.Width <= 0 || cropBounds.Height <= 0)
            return full;

        using var crop = new Mat(source, cropBounds);
        using var gray = new Mat();
        using var black = new Mat();
        using var white = new Mat();
        using var local = Mat.Zeros(
            cropBounds.Height,
            cropBounds.Width,
            MatType.CV_8UC1).ToMat();

        Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);

        Cv2.Threshold(
            gray,
            black,
            0,
            255,
            ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

        Cv2.Threshold(
            gray,
            white,
            0,
            255,
            ThresholdTypes.Binary | ThresholdTypes.Otsu);

        AddTextSizedComponents(black, local);
        AddTextSizedComponents(white, local);

        // Restrict the result back to the detector's TextBubble extent.
        var textLocal = new Rect(
            Math.Max(0, textBounds.X - cropBounds.X),
            Math.Max(0, textBounds.Y - cropBounds.Y),
            Math.Min(textBounds.Right, cropBounds.Right) -
                Math.Max(textBounds.Left, cropBounds.Left),
            Math.Min(textBounds.Bottom, cropBounds.Bottom) -
                Math.Max(textBounds.Top, cropBounds.Top));

        using var restricted = Mat.Zeros(
            local.Rows,
            local.Cols,
            MatType.CV_8UC1).ToMat();

        if (textLocal.Width > 0 && textLocal.Height > 0)
        {
            using var srcRoi = new Mat(local, textLocal);
            using var dstRoi = new Mat(restricted, textLocal);
            srcRoi.CopyTo(dstRoi);
        }

        using (var kernel = Cv2.GetStructuringElement(
            MorphShapes.Ellipse,
            new Size(3, 3)))
        {
            Cv2.Dilate(
                restricted,
                restricted,
                kernel,
                iterations: 1);
        }

        using var destination = new Mat(full, cropBounds);
        restricted.CopyTo(destination);

        return full;
    }

    static void AddTextSizedComponents(
        Mat binary,
        Mat destination)
    {
        Cv2.FindContours(
            binary,
            out Point[][] contours,
            out _,
            RetrievalModes.External,
            ContourApproximationModes.ApproxSimple);

        double cropArea =
            Math.Max(1.0, binary.Rows * (double)binary.Cols);

        foreach (var contour in contours)
        {
            if (contour.Length == 0)
                continue;

            var bounds = Cv2.BoundingRect(contour);
            double area = Math.Abs(Cv2.ContourArea(contour));

            bool ordinary = area > 10;
            bool punctuation =
                area >= 4 &&
                bounds.Width <= 6 &&
                bounds.Height <= 6;

            if (!ordinary && !punctuation)
                continue;

            const int margin = 1;
            bool touchesBorder =
                bounds.Left < margin ||
                bounds.Top < margin ||
                bounds.Right > binary.Cols - margin ||
                bounds.Bottom > binary.Rows - margin;

            if (touchesBorder)
                continue;

            // A component occupying half the crop is normally the bubble or
            // narration-box background, not lettering.
            if (area >= cropArea * 0.50)
                continue;

            Cv2.DrawContours(
                destination,
                [contour],
                -1,
                Scalar.White,
                thickness: -1);
        }
    }

    static Rect ClampRect(
        int x,
        int y,
        int width,
        int height,
        int imageWidth,
        int imageHeight)
    {
        int left = Math.Clamp(x, 0, imageWidth);
        int top = Math.Clamp(y, 0, imageHeight);
        int right = Math.Clamp(x + Math.Max(0, width), 0, imageWidth);
        int bottom = Math.Clamp(y + Math.Max(0, height), 0, imageHeight);

        return new Rect(
            left,
            top,
            Math.Max(0, right - left),
            Math.Max(0, bottom - top));
    }

    static Rect Intersect(Rect a, Rect b)
    {
        int left = Math.Max(a.Left, b.Left);
        int top = Math.Max(a.Top, b.Top);
        int right = Math.Min(a.Right, b.Right);
        int bottom = Math.Min(a.Bottom, b.Bottom);

        return new Rect(
            left,
            top,
            Math.Max(0, right - left),
            Math.Max(0, bottom - top));
    }
}
