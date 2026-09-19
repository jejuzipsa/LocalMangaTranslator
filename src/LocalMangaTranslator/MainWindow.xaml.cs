using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Win32;
using LocalMangaTranslator.Models;
using LocalMangaTranslator.Services;

namespace LocalMangaTranslator;

public partial class MainWindow : System.Windows.Window
{
    const int DwmwaUseImmersiveDarkMode = 20;
    const int DwmwaUseImmersiveDarkModeBefore20H1 = 19;
    const int DwmwaCaptionColor = 35;
    const int DwmwaTextColor = 36;

    [DllImport("dwmapi.dll")]
    static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute, ref int value, int size);

    static readonly int DarkCaptionColor = ColorRef(0x0D, 0x11, 0x17);
    static readonly int LightTextColor = ColorRef(0xE6, 0xED, 0xF3);

    static int ColorRef(byte r, byte g, byte b) => r | (g << 8) | (b << 16);
    static readonly HashSet<string> ImageExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".webp", ".bmp" };

    readonly OllamaClient ollama = new();
    readonly VisionTranslationService vision = new();
    readonly TranslationRefinementService translationRefiner = new();
    readonly OcrBlockGrouper blockGrouper = new();
    readonly OcrContainerUnitBuilder unitBuilder = new();
    readonly RenderPipelineService renderer = new();

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

        SourceInitialized += (_, _) => ApplyDarkTitleBar();

        LoadModels();
        Log("프로그램 시작");
        InitializeOcr();
        Loaded += async (_, _) => await CheckSelectedModelsAsync();
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
        var reviewModels = catalog.LoadForTask("review")
            .Where(x => x.SupportsImage)
            .ToList();
        var translationModels = catalog.LoadForTask("translation")
            .Where(x => x.SupportsText)
            .ToList();

        ReviewModelBox.ItemsSource = reviewModels;
        TranslationModelBox.ItemsSource = translationModels;

        SelectPreferredModel(ReviewModelBox, reviewModels, "gemma4-12b");
        SelectPreferredModel(TranslationModelBox, translationModels, "gemma4-12b");

        Log($"모델 프로필 · OCR 검수 {reviewModels.Count}개 / 번역 {translationModels.Count}개");
        Log($"OCR 엔진 · {ocr?.Status ?? "초기화 전"}");
    }

    static void SelectPreferredModel(
        System.Windows.Controls.ComboBox box,
        IReadOnlyList<ModelProfile> models,
        string preferredId)
    {
        if (models.Count == 0) return;

        var preferred = models.FirstOrDefault(x =>
            string.Equals(x.Id, preferredId, StringComparison.OrdinalIgnoreCase));

        box.SelectedItem = preferred ?? models[0];
    }

    void ApplyDarkTitleBar()
    {
        try
        {
            var hwnd = new System.Windows.Interop.WindowInteropHelper(this).Handle;
            if (hwnd == IntPtr.Zero) return;

            int enabled = 1;
            if (DwmSetWindowAttribute(hwnd, DwmwaUseImmersiveDarkMode, ref enabled, sizeof(int)) != 0)
                DwmSetWindowAttribute(hwnd, DwmwaUseImmersiveDarkModeBefore20H1, ref enabled, sizeof(int));

            int caption = DarkCaptionColor;
            DwmSetWindowAttribute(hwnd, DwmwaCaptionColor, ref caption, sizeof(int));

            int textColor = LightTextColor;
            DwmSetWindowAttribute(hwnd, DwmwaTextColor, ref textColor, sizeof(int));
        }
        catch
        {
            // 구형 Windows에서는 제목 표시줄 색상 변경을 건너뛴다.
        }
    }

    void ModelBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (LogBox is null) return;

        if (sender == ReviewModelBox && ReviewModelBox.SelectedItem is ModelProfile review)
            Log($"OCR 검수 모델 선택: {review.Name} | {review.ModelTag}");
        else if (sender == TranslationModelBox && TranslationModelBox.SelectedItem is ModelProfile translation)
            Log($"번역 모델 선택: {translation.Name} | {translation.ModelTag}");
    }

    async Task CheckSelectedModelsAsync()
    {
        var selected = new[]
        {
            ReviewModelBox.SelectedItem as ModelProfile,
            TranslationModelBox.SelectedItem as ModelProfile
        }
        .Where(x => x is not null)
        .Cast<ModelProfile>()
        .GroupBy(x => (x.ApiBase, x.ModelTag))
        .Select(x => x.First())
        .ToList();

        foreach (var model in selected)
        {
            if (!await ollama.IsServerReadyAsync(model))
            {
                Log("Ollama 서버를 찾지 못했습니다 · [선택 모델 확인/설치]에서 Ollama와 모델을 준비할 수 있습니다");
                return;
            }

            var installed = await ollama.IsModelInstalledAsync(model);
            Log(installed
                ? $"모델 준비됨: {model.ModelTag}"
                : $"모델 미설치: {model.ModelTag} · [선택 모델 확인/설치]로 다운로드 가능");
        }
    }

    async void InstallModel_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        var selected = new[]
        {
            ReviewModelBox.SelectedItem as ModelProfile,
            TranslationModelBox.SelectedItem as ModelProfile
        }
        .Where(x => x is not null)
        .Cast<ModelProfile>()
        .GroupBy(x => (x.ApiBase, x.ModelTag))
        .Select(x => x.First())
        .ToList();

        if (selected.Count == 0)
        {
            Log("확인/설치할 모델을 선택하세요");
            return;
        }

        InstallModelButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        workCts = new CancellationTokenSource();

        try
        {
            var runtimeModel = selected[0];
            if (!await ollama.IsServerReadyAsync(runtimeModel, workCts.Token))
            {
                var ready = await EnsureOllamaRuntimeAsync(runtimeModel, workCts.Token);
                if (!ready) return;
            }

            foreach (var model in selected)
            {
                if (await ollama.IsModelInstalledAsync(model, workCts.Token))
                {
                    Log($"이미 설치되어 있습니다: {model.ModelTag}");
                    continue;
                }

                Log($"모델 다운로드 시작: {model.ModelTag}");
                var progress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"모델 다운로드 · {model.Name} · {message}";
                    Log($"모델 다운로드 · {model.ModelTag} · {message}");
                });

                await ollama.PullModelAsync(model, progress, workCts.Token);
                Log($"모델 다운로드 완료: {model.ModelTag}");
            }

            CurrentStatusText.Text = "선택 모델 준비 완료";
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
            "로컬 AI 모델을 실행하려면 Ollama가 필요합니다.\n\n지금 Ollama를 자동 설치할까요?",
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

        Log("Ollama 런타임 설치 완료 · 서버 시작 확인 중");
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

        if (ReviewModelBox.SelectedItem is not ModelProfile reviewModel)
        {
            Log("OCR 검수 모델을 선택해야 합니다");
            return;
        }

        if (TranslationModelBox.SelectedItem is not ModelProfile translationModel)
        {
            Log("번역 모델을 선택해야 합니다");
            return;
        }

        if (ocr is null || !ocr.Ready)
        {
            Log($"OCR을 사용할 수 없습니다: {ocr?.Status ?? "초기화 안됨"}");
            return;
        }

        if (!await ollama.IsServerReadyAsync(reviewModel) ||
            !await ollama.IsServerReadyAsync(translationModel))
        {
            Log("Ollama 서버에 연결할 수 없습니다");
            return;
        }

        foreach (var requiredModel in new[] { reviewModel, translationModel }
                     .GroupBy(x => (x.ApiBase, x.ModelTag))
                     .Select(x => x.First()))
        {
            if (!await ollama.IsModelInstalledAsync(requiredModel))
            {
                Log($"선택 모델이 설치되어 있지 않습니다: {requiredModel.ModelTag}");
                return;
            }
        }

        Directory.CreateDirectory(OutputPathBox.Text);
        workCts = new CancellationTokenSource();
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        InstallModelButton.IsEnabled = false;

        Log($"작업 시작 | OCR: {ocr.Status} | 검수: {reviewModel.Name} | 번역: {translationModel.Name}");

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

                item.Status = "컨테이너 단위 구성";
                SetStatus(i, item, "OCR 컨테이너 / line ownership");

                var preliminaryBlocks = blockGrouper.Group(lines);

                var unitBuild = await Task.Run(
                    () => unitBuilder.Build(
                        item.FilePath,
                        lines,
                        preliminaryBlocks,
                        workCts.Token),
                    workCts.Token);

                var blocks = unitBuild.Units.ToList();

                Log(
                    $"{item.FileName} | OCR unit 구성 · " +
                    $"{lines.Count}줄 → 예비 {preliminaryBlocks.Count}블록 → " +
                    $"컨테이너 {unitBuild.ContainerCount}개 / " +
                    $"배정 {unitBuild.AssignedLineCount}줄 / " +
                    $"고아 {unitBuild.OrphanGroupCount}블록 → " +
                    $"Vision 입력 {blocks.Count} unit");

                item.Status = "OCR 검수";
                SetStatus(i, item, $"Vision OCR 검수 · {reviewModel.Name}");
                Log($"{item.FileName} | Vision OCR 검수 시작 · {reviewModel.ModelTag}");

                var visionProgress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"{i + 1}/{Queue.Count} · {item.FileName} · {message}";
                    Log($"{item.FileName} | {message}");
                });

                var reviewed = await vision.ReviewAndTranslateAsync(
                    item.FilePath,
                    blocks,
                    reviewModel,
                    visionProgress,
                    workCts.Token);

                int corrected = reviewed.Count(x =>
                    !string.Equals(x.Source.Text.Trim(), x.CorrectedText.Trim(), StringComparison.Ordinal));

                int skipped = reviewed.Count(x => !x.Render);
                Log($"{item.FileName} | Vision 검수 완료 · {reviewed.Count}개 블록 · OCR 교정 {corrected}개 · 조판 제외 {skipped}개");

                item.Status = "번역 중";
                SetStatus(i, item, $"최종 번역 · {translationModel.Name}");
                Log($"{item.FileName} | 최종 번역 시작 · {translationModel.ModelTag}");

                var translationProgress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"{i + 1}/{Queue.Count} · {item.FileName} · {message}";
                    Log($"{item.FileName} | {message}");
                });

                var translated = await translationRefiner.TranslateAsync(
                    reviewed,
                    translationModel,
                    translationProgress,
                    workCts.Token);

                Log($"{item.FileName} | 최종 번역 완료 · {translated.Count(x => x.Render)}개 조판 대상");

                var document = new VisionTranslationDocument
                {
                    SourceFile = item.FileName,
                    Model = $"review={reviewModel.ModelTag}; translation={translationModel.ModelTag}",
                    Regions = translated
                };

                var jsonPath = Path.Combine(
                    OutputPathBox.Text,
                    Path.GetFileNameWithoutExtension(item.FilePath) + ".translation.json");

                await File.WriteAllTextAsync(
                    jsonPath,
                    JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true }),
                    workCts.Token);

                Log($"{item.FileName} | 번역 JSON 저장: {Path.GetFileName(jsonPath)}");

                item.Status = "인페인트 중";
                SetStatus(i, item, "인페인트");
                Log($"{item.FileName} | 인페인트 시작");

                var imagePath = Path.Combine(
                    OutputPathBox.Text,
                    Path.GetFileNameWithoutExtension(item.FilePath) + ".translated.png");

                var renderProgress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"{i + 1}/{Queue.Count} · {item.FileName} · {message}";
                    Log($"{item.FileName} | {message}");
                });

                await renderer.RenderAsync(
                    item.FilePath,
                    translated,
                    imagePath,
                    renderProgress,
                    workCts.Token);

                item.Status = "완료";
                Log($"{item.FileName} | 완료 이미지: {Path.GetFileName(imagePath)}");

                FinishItem(i, item, sw);
            }

            CurrentStatusText.Text = "번역 이미지 생성 완료";
            Log("전체 작업 완료 · OCR → Vision OCR 검수 → 번역 → 인페인트 → 한글 조판");
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
