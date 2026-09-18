using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace LocalMangaTranslator.Models;

public sealed class QueueItem : INotifyPropertyChanged
{
    string status = "대기";
    string elapsed = "-";

    public required string FilePath { get; init; }
    public string FileName => Path.GetFileName(FilePath);

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

    public event PropertyChangedEventHandler? PropertyChanged;
    void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
