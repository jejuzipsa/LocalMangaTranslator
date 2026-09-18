using System.Collections.ObjectModel;
using System.Diagnostics;
using Microsoft.Win32;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.Services;

namespace LocalMangaTranslator;

public partial class MainWindow
{
    static readonly HashSet<string> ImageExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".webp", ".bmp" };

    CancellationTokenSource? workCts;
    OcrEngine? ocr;

    public ObservableCollection<QueueItem> Queue { get; } = [];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        var defaultOutput = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
            "LocalMangaTranslator_Output");
        OutputPathBox.Text = defaultOutput;

        LoadModels();
        Log("프로그램 시작");
        InitializeOcr();
    }

    void InitializeOcr()
    {
        try
        {
            var modelRoot = Path.Combine(AppContext.BaseDirectory, "models", "ocr");
            ocr = new OcrEngine(modelRoot);
            Log($"OCR: {ocr.Status}");
        }
        catch (Exception ex)
        {
            Log($"OCR 초기화 실패: {ex.Message}");
        }
    }

    void LoadModels()
    {
        var root = Path.Combine(AppContext.BaseDirectory, "models", "vision_llm");
        var catalog = new ModelCatalog(root);
        var models = catalog.Load();

        ModelBox.ItemsSource = models;
        if (models.Count > 0) ModelBox.SelectedIndex = 0;
        Log(models.Count > 0
            ? $"Vision 모델 프로필 {models.Count}개 발견"
            : "Vision 모델 프로필이 없습니다");
    }

    void AddFiles(IEnumerable<string> paths)
    {
        var existing = Queue.Select(x => x.FilePath).ToHashSet(StringComparer.OrdinalIgnoreCase);

        foreach (var path in paths)
        {
            if (Directory.Exists(path))
            {
                AddFolder(path);
                continue;
            }

            if (!File.Exists(path) || !ImageExtensions.Contains(Path.GetExtension(path)) || existing.Contains(path))
                continue;

            Queue.Add(new QueueItem { FilePath = path });
            existing.Add(path);
        }

        QueueCountText.Text = $"{Queue.Count}개";
    }

    void AddFolder(string folder)
    {
        try
        {
            var files = Directory.EnumerateFiles(folder, "*.*", SearchOption.TopDirectoryOnly)
                .Where(x => ImageExtensions.Contains(Path.GetExtension(x)))
                .OrderBy(x => x, StringComparer.CurrentCultureIgnoreCase);
            AddFiles(files);
        }
        catch (Exception ex)
        {
            Log($"폴더 읽기 실패: {ex.Message}");
        }
    }

    void AddFiles_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Multiselect = true,
            Filter = "이미지|*.jpg;*.jpeg;*.png;*.webp;*.bmp|모든 파일|*.*"
        };
        if (dialog.ShowDialog() == true) AddFiles(dialog.FileNames);
    }

    void AddFolder_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "이미지 폴더 선택" };
        if (dialog.ShowDialog() == true) AddFolder(dialog.FolderName);
    }

    void Clear_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        if (workCts is not null) return;
        Queue.Clear();
        QueueCountText.Text = "0개";
        OverallProgress.Value = 0;
        CurrentStatusText.Text = "대기 중";
        Log("작업 목록 전체 삭제");
    }

    void ChooseOutput_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "번역 결과 저장 폴더",
            InitialDirectory = OutputPathBox.Text
        };
        if (dialog.ShowDialog() == true)
            OutputPathBox.Text = dialog.FolderName;
    }

    void OpenOutput_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        try
        {
            Directory.CreateDirectory(OutputPathBox.Text);
            Process.Start(new ProcessStartInfo("explorer.exe", OutputPathBox.Text) { UseShellExecute = true });
        }
        catch (Exception ex)
        {
            Log($"폴더 열기 실패: {ex.Message}");
        }
    }

    async void Start_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        if (Queue.Count == 0)
        {
            Log("작업할 이미지가 없습니다");
            return;
        }

        if (ModelBox.SelectedItem is not ModelProfile model)
        {
            Log("Vision 모델을 선택해야 합니다");
            return;
        }

        if (ocr is null || !ocr.Ready)
        {
            Log($"OCR을 사용할 수 없습니다: {ocr?.Status ?? "초기화 안됨"}");
            return;
        }

        Directory.CreateDirectory(OutputPathBox.Text);
        workCts = new CancellationTokenSource();
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = true;

        Log($"작업 시작 | 모델: {model.Name}");

        try
        {
            for (int i = 0; i < Queue.Count; i++)
            {
                workCts.Token.ThrowIfCancellationRequested();
                var item = Queue[i];
                var sw = Stopwatch.StartNew();

                item.Status = "OCR 중";
                CurrentStatusText.Text = $"{i + 1}/{Queue.Count} · {item.FileName} · OCR";
                Log($"{item.FileName} | OCR 시작");

                var lines = await ocr.RecognizeAsync(item.FilePath, workCts.Token);
                Log($"{item.FileName} | OCR 완료 · {lines.Count}개 영역");

                // 다음 커밋에서 이 지점에 Vision LLM 검수+번역을 연결한다.
                item.Status = "Vision 연결 대기";
                Log($"{item.FileName} | OCR 결과 준비 완료 · Vision LLM 단계는 아직 미연결");

                sw.Stop();
                item.Elapsed = $"{sw.Elapsed.TotalSeconds:0.0}초";
                OverallProgress.Value = (i + 1) * 100.0 / Queue.Count;
            }

            CurrentStatusText.Text = "OCR 단계 완료 · Vision LLM 연결 필요";
            Log("현재 구현 범위 완료: UI + 작업 큐 + OCR");
        }
        catch (OperationCanceledException)
        {
            CurrentStatusText.Text = "작업 중지됨";
            Log("사용자가 작업을 중지했습니다");
        }
        catch (Exception ex)
        {
            CurrentStatusText.Text = "오류";
            Log($"작업 오류: {ex}");
        }
        finally
        {
            workCts.Dispose();
            workCts = null;
            StartButton.IsEnabled = true;
            StopButton.IsEnabled = false;
        }
    }

    void Stop_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        if (workCts is null) return;
        Log("작업 중지 요청");
        workCts.Cancel();
    }

    void Window_Drop(object sender, System.Windows.DragEventArgs e)
    {
        if (e.Data.GetDataPresent(System.Windows.DataFormats.FileDrop) &&
            e.Data.GetData(System.Windows.DataFormats.FileDrop) is string[] paths)
            AddFiles(paths);
    }

    void ClearLog_Click(object sender, System.Windows.RoutedEventArgs e) => LogBox.Clear();

    void Log(string message)
    {
        if (LogBox is null) return;
        LogBox.AppendText($"[{DateTime.Now:HH:mm:ss}] {message}{Environment.NewLine}");
        LogBox.ScrollToEnd();
    }

    protected override void OnClosed(EventArgs e)
    {
        workCts?.Cancel();
        workCts?.Dispose();
        ocr?.Dispose();
        base.OnClosed(e);
    }
}
