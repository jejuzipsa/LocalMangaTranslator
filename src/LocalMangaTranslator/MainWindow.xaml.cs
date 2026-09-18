using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Win32;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.Services;

namespace LocalMangaTranslator;

public partial class MainWindow : System.Windows.Window
{
    static readonly HashSet<string> ImageExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".webp", ".bmp" };

    readonly OllamaClient ollama = new();
    readonly VisionTranslationService vision = new();

    CancellationTokenSource? workCts;
    OcrEngine? ocr;

    public ObservableCollection<QueueItem> Queue { get; } = [];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        OutputPathBox.Text = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyPictures),
            "LocalMangaTranslator_Output");

        LoadModels();
        Log("프로그램 시작");
        InitializeOcr();
        Loaded += async (_, _) => await CheckSelectedModelAsync();
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
        var models = new ModelCatalog(root).Load();
        ModelBox.ItemsSource = models;
        if (models.Count > 0) ModelBox.SelectedIndex = 0;

        Log(models.Count > 0
            ? $"Vision 모델 프로필 {models.Count}개 발견"
            : "Vision 모델 프로필이 없습니다");
    }

    async Task CheckSelectedModelAsync()
    {
        if (ModelBox.SelectedItem is not ModelProfile model) return;

        if (!await ollama.IsServerReadyAsync(model))
        {
            Log("Ollama 서버를 찾지 못했습니다 · [모델 확인/설치] 버튼에서 Ollama와 모델을 준비할 수 있습니다");
            return;
        }

        var installed = await ollama.IsModelInstalledAsync(model);
        Log(installed
            ? $"모델 준비됨: {model.ModelTag}"
            : $"모델 미설치: {model.ModelTag} · [모델 확인/설치] 버튼으로 다운로드 가능");
    }

    async void InstallModel_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        if (ModelBox.SelectedItem is not ModelProfile model)
        {
            Log("설치할 Vision 모델을 선택하세요");
            return;
        }

        InstallModelButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        workCts = new CancellationTokenSource();

        try
        {
            if (!await ollama.IsServerReadyAsync(model, workCts.Token))
            {
                var ready = await EnsureOllamaRuntimeAsync(model, workCts.Token);
                if (!ready) return;
            }

            if (await ollama.IsModelInstalledAsync(model, workCts.Token))
            {
                Log($"이미 설치되어 있습니다: {model.ModelTag}");
                CurrentStatusText.Text = "모델 준비 완료";
                return;
            }

            Log($"모델 다운로드 시작: {model.ModelTag}");
            var progress = new Progress<string>(message =>
            {
                CurrentStatusText.Text = $"모델 다운로드 · {message}";
                Log($"모델 다운로드 · {message}");
            });

            await ollama.PullModelAsync(model, progress, workCts.Token);
            Log($"모델 다운로드 완료: {model.ModelTag}");
            CurrentStatusText.Text = "모델 준비 완료";
        }
        catch (OperationCanceledException)
        {
            Log("설치/다운로드 중지됨");
            CurrentStatusText.Text = "작업 중지됨";
        }
        catch (Exception ex)
        {
            Log($"모델 준비 실패: {ex.Message}");
            CurrentStatusText.Text = "모델 준비 실패";
        }
        finally
        {
            workCts.Dispose();
            workCts = null;
            InstallModelButton.IsEnabled = true;
            StopButton.IsEnabled = false;
        }
    }

    async Task<bool> EnsureOllamaRuntimeAsync(ModelProfile model, CancellationToken token)
    {
        var answer = System.Windows.MessageBox.Show(
            "Qwen Vision 모델을 실행하려면 Ollama가 필요합니다.\n\n지금 Ollama를 자동 설치할까요?",
            "Ollama 설치",
            System.Windows.MessageBoxButton.YesNo,
            System.Windows.MessageBoxImage.Question);

        if (answer != System.Windows.MessageBoxResult.Yes)
        {
            Log("Ollama 설치 취소");
            CurrentStatusText.Text = "Ollama 설치 필요";
            return false;
        }

        Log("Ollama 설치 시작 · Windows 패키지 관리자(winget) 사용");
        CurrentStatusText.Text = "Ollama 설치 중...";

        var psi = new ProcessStartInfo
        {
            FileName = "winget.exe",
            Arguments = "install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements --disable-interactivity",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("winget을 실행할 수 없습니다.");

        var stdoutTask = process.StandardOutput.ReadToEndAsync(token);
        var stderrTask = process.StandardError.ReadToEndAsync(token);
        await process.WaitForExitAsync(token);

        var stdout = await stdoutTask;
        var stderr = await stderrTask;

        if (process.ExitCode != 0)
        {
            var detail = string.IsNullOrWhiteSpace(stderr) ? stdout : stderr;
            detail = detail.Replace('\r', ' ').Replace('\n', ' ').Trim();
            if (detail.Length > 300) detail = detail[..300] + "...";
            throw new InvalidOperationException(
                $"Ollama 설치 실패 (winget 종료코드 {process.ExitCode}) · {detail}");
        }

        Log("Ollama 설치 완료 · 서버 시작 확인 중");
        CurrentStatusText.Text = "Ollama 시작 확인 중...";

        if (await WaitForOllamaAsync(model, token, 12))
        {
            Log("Ollama 서버 준비 완료");
            return true;
        }

        var exe = FindOllamaExe();
        if (exe is not null)
        {
            Log($"Ollama 서버 시작: {exe}");
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = exe,
                    Arguments = "serve",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden
                });
            }
            catch (Exception ex)
            {
                Log($"Ollama 서버 시작 재시도 실패: {ex.Message}");
            }
        }
        else
        {
            Log("Ollama 실행 파일을 찾지 못했습니다 · 설치 후 프로그램 재실행이 필요할 수 있습니다");
        }

        if (await WaitForOllamaAsync(model, token, 20))
        {
            Log("Ollama 서버 준비 완료");
            return true;
        }

        throw new InvalidOperationException(
            "Ollama는 설치됐지만 서버가 시작되지 않았습니다. 프로그램을 한 번 다시 실행해 주세요.");
    }

    async Task<bool> WaitForOllamaAsync(ModelProfile model, CancellationToken token, int seconds)
    {
        for (int i = 0; i < seconds; i++)
        {
            token.ThrowIfCancellationRequested();
            if (await ollama.IsServerReadyAsync(model, token))
                return true;

            await Task.Delay(1000, token);
        }

        return false;
    }

    static string? FindOllamaExe()
    {
        var candidates = new[]
        {
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs", "Ollama", "ollama.exe"),
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Ollama", "ollama.exe"),
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "Ollama", "ollama.exe")
        };

        foreach (var path in candidates)
            if (File.Exists(path))
                return path;

        var pathValue = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in pathValue.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                var candidate = Path.Combine(dir.Trim(), "ollama.exe");
                if (File.Exists(candidate)) return candidate;
            }
            catch { }
        }

        return null;
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

            if (!File.Exists(path) ||
                !ImageExtensions.Contains(Path.GetExtension(path)) ||
                existing.Contains(path))
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
            Process.Start(new ProcessStartInfo("explorer.exe", OutputPathBox.Text)
            {
                UseShellExecute = true
            });
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

        if (!await ollama.IsServerReadyAsync(model))
        {
            Log("Ollama 서버에 연결할 수 없습니다");
            return;
        }

        if (!await ollama.IsModelInstalledAsync(model))
        {
            Log($"Vision 모델이 설치되어 있지 않습니다: {model.ModelTag}");
            return;
        }

        Directory.CreateDirectory(OutputPathBox.Text);
        workCts = new CancellationTokenSource();
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        InstallModelButton.IsEnabled = false;

        Log($"작업 시작 | 모델: {model.Name}");

        try
        {
            for (int i = 0; i < Queue.Count; i++)
            {
                workCts.Token.ThrowIfCancellationRequested();
                var item = Queue[i];
                var sw = Stopwatch.StartNew();

                item.Status = "OCR 중";
                SetStatus(i, item, "OCR");
                Log($"{item.FileName} | OCR 시작");

                var lines = await ocr.RecognizeAsync(item.FilePath, workCts.Token);
                Log($"{item.FileName} | OCR 완료 · {lines.Count}개 영역");

                if (lines.Count == 0)
                {
                    item.Status = "텍스트 없음";
                    Log($"{item.FileName} | 감지된 텍스트 없음");
                    FinishItem(i, item, sw);
                    continue;
                }

                item.Status = "OCR 검수·번역";
                SetStatus(i, item, "Qwen Vision OCR 검수·번역");
                Log($"{item.FileName} | Vision LLM 검수·번역 시작");

                var translated = await vision.ReviewAndTranslateAsync(
                    item.FilePath,
                    lines,
                    model,
                    workCts.Token);

                int corrected = translated.Count(x =>
                    !string.Equals(x.Source.Text.Trim(), x.CorrectedText.Trim(), StringComparison.Ordinal));

                Log($"{item.FileName} | Vision 완료 · {translated.Count}개 번역 · OCR 교정 {corrected}개");

                var document = new VisionTranslationDocument
                {
                    SourceFile = item.FileName,
                    Model = model.ModelTag,
                    Regions = translated
                };

                var jsonPath = Path.Combine(
                    OutputPathBox.Text,
                    Path.GetFileNameWithoutExtension(item.FilePath) + ".translation.json");

                await File.WriteAllTextAsync(
                    jsonPath,
                    JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true }),
                    workCts.Token);

                item.Status = "번역 데이터 완료";
                Log($"{item.FileName} | 번역 JSON 저장: {Path.GetFileName(jsonPath)}");
                Log($"{item.FileName} | 다음 단계: 인페인트/조판 연결 예정");

                FinishItem(i, item, sw);
            }

            CurrentStatusText.Text = "OCR + Vision 검수·번역 완료";
            Log("현재 구현 범위 완료 · 다음 단계는 인페인트와 한글 조판");
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
            InstallModelButton.IsEnabled = true;
        }
    }

    void SetStatus(int index, QueueItem item, string stage)
        => CurrentStatusText.Text = $"{index + 1}/{Queue.Count} · {item.FileName} · {stage}";

    void FinishItem(int index, QueueItem item, Stopwatch sw)
    {
        sw.Stop();
        item.Elapsed = $"{sw.Elapsed.TotalSeconds:0.0}초";
        OverallProgress.Value = (index + 1) * 100.0 / Queue.Count;
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
