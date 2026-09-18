using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace LocalMangaTranslator.Models;

public sealed class QueueItem : INotifyPropertyChanged
{
    string status = "대기";
    string elapsed = "-";
    ImageSource? preview;

    public required string FilePath { get; init; }
    public string FileName => Path.GetFileName(FilePath);

    public ImageSource? Preview
    {
        get
        {
            preview ??= LoadPreview(FilePath);
            return preview;
        }
    }

    public string Status
    {
        get => status;
        set { status = value; OnPropertyChanged(); }
    }

    public string Elapsed
    {
        get => elapsed;
        set { elapsed = value; OnPropertyChanged(); }
    }

    static ImageSource? LoadPreview(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.DecodePixelWidth = 120;
            bitmap.StreamSource = stream;
            bitmap.EndInit();
            bitmap.Freeze();
            return bitmap;
        }
        catch
        {
            return null;
        }
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
