using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
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
    readonly RenderPipelineService renderer = new();

    CancellationTokenSource? workCts;
    OcrEngine? ocr;
    BaberuOcrEngine? baberu;
    PageAnalysisService? pageAnalysis;
    OcrPipelineService? ocrPipeline;
    PagePipelineService? pagePipeline;

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
            baberu = new BaberuOcrEngine();
            pageAnalysis = new PageAnalysisService();
            ocrPipeline = new OcrPipelineService(ocr, baberu);
            pagePipeline = new PagePipelineService(
                pageAnalysis,
                ocrPipeline,
                vision,
                translationRefiner,
                renderer);
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
        Log(ExternalModelManager.IsRtdetrReady()
            ? "페이지 분석 모델 · RT-DETR 준비됨"
            : "페이지 분석 모델 · RT-DETR 미설치");
        Log(ExternalModelManager.IsBaberuReady()
            ? "보조 OCR · Baberu 준비됨"
            : "보조 OCR · Baberu 미설치");
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

            if (HybridPageAnalysisCheckBox.IsChecked == true)
            {
                var layoutProgress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"페이지 분석 모델 · {message}";
                    Log($"페이지 분석 모델 · {message}");
                });

                await ExternalModelManager.EnsureRtdetrAsync(
                    layoutProgress,
                    workCts.Token);
            }

            if (BaberuOcrCheckBox.IsChecked == true)
            {
                var baberuProgress = new Progress<string>(message =>
                {
                    CurrentStatusText.Text = $"Baberu OCR · {message}";
                    Log($"Baberu OCR · {message}");
                });

                await ExternalModelManager.EnsureBaberuAsync(
                    baberuProgress,
                    workCts.Token);
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

        if (ocr is null ||
            !ocr.Ready ||
            ocrPipeline is null ||
            pagePipeline is null)
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

        if (HybridPageAnalysisCheckBox.IsChecked == true &&
            !ExternalModelManager.IsRtdetrReady())
        {
            Log("RT-DETR 페이지 분석 모델이 없습니다 · [선택 모델 확인/설치]을 눌러 설치해 주세요");
            return;
        }

        if (BaberuOcrCheckBox.IsChecked == true &&
            !ExternalModelManager.IsBaberuReady())
        {
            Log("Baberu 보조 OCR 모델이 없습니다 · [선택 모델 확인/설치]을 눌러 설치해 주세요");
            return;
        }

        var pipelineOptions = new PipelineOptions(
            HybridPageAnalysisCheckBox.IsChecked == true
                ? RegionAnalysisMode.HybridRtdetr
                : RegionAnalysisMode.Legacy,
            TargetedRegionOcrCheckBox.IsChecked == true,
            BaberuOcrCheckBox.IsChecked == true,
            FinalAuditCheckBox.IsChecked == true);

        Directory.CreateDirectory(OutputPathBox.Text);
        workCts = new CancellationTokenSource();
        StartButton.IsEnabled = false;
        StopButton.IsEnabled = true;
        InstallModelButton.IsEnabled = false;

        Log($"작업 시작 | 페이지 분석: {pipelineOptions.RegionAnalysis} | Baberu: {pipelineOptions.EnableBaberuOcr} | 영역 OCR 진단: {pipelineOptions.EnableTargetedRegionOcr} | 완료검토: {pipelineOptions.EnableFinalAudit} | 판단트랙: {pipelineOptions.DecisionTrack}/{pipelineOptions.LayaMode} | OCR: {ocr.Status} | 검수: {reviewModel.Name} | 번역: {translationModel.Name}");

        try
        {
            for (int i = 0; i < Queue.Count; i++)
            {
                workCts.Token.ThrowIfCancellationRequested();
                var item = Queue[i];
                var sw = Stopwatch.StartNew();

                item.Status = "파이프라인";
                SetStatus(i, item, "페이지 파이프라인 시작");
                Log($"{item.FileName} | 페이지 파이프라인 시작");

                var pipelineProgress =
                    new Progress<PipelineProgress>(update =>
                    {
                        item.Status =
                            PipelineStageLabel(update.Stage);

                        CurrentStatusText.Text =
                            $"{i + 1}/{Queue.Count} · {item.FileName} · {update.Message}";

                        Log(
                            $"{item.FileName} | " +
                            $"[{PipelineStageLabel(update.Stage)}] " +
                            update.Message);
                    });

                var result =
                    await pagePipeline.ProcessAsync(
                        item.FilePath,
                        OutputPathBox.Text,
                        reviewModel,
                        translationModel,
                        pipelineOptions,
                        pipelineProgress,
                        workCts.Token);

                if (!result.HasText)
                {
                    item.Status = "텍스트 없음";
                    Log($"{item.FileName} | 감지된 텍스트 없음");
                    FinishItem(i, item, sw);
                    continue;
                }

                item.Status = "완료";
                FinishItem(i, item, sw);
            }

            CurrentStatusText.Text = "번역 이미지 생성 완료";
            Log("전체 작업 완료 · 페이지 분석 + OCR → Evidence Fusion → Vision 검수 → 번역 → 안전 삭제/조판");
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

    static string PipelineStageLabel(
        PipelineStageKind stage)
        => stage switch
        {
            PipelineStageKind.PageAnalysis => "페이지 분석",
            PipelineStageKind.ContainerDetection => "컨테이너 검출",
            PipelineStageKind.RegionOcr => "영역 OCR",
            PipelineStageKind.ContainerValidation => "컨테이너 검증",
            PipelineStageKind.ContainerOcr => "컨테이너 OCR",
            PipelineStageKind.OcrValidation => "OCR 검증",
            PipelineStageKind.OcrObservation => "OCR 관측",
            PipelineStageKind.OcrUnitFormation => "OCR Unit",
            PipelineStageKind.VisionReview => "Vision 검수",
            PipelineStageKind.Translation => "번역",
            PipelineStageKind.Render => "삭제/조판",
            PipelineStageKind.FinalAudit => "완료 검토",
            PipelineStageKind.Completed => "완료",
            _ => stage.ToString()
        };

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
        pageAnalysis?.Dispose();
        baberu?.Dispose();
        ocr?.Dispose();
        base.OnClosed(e);
    }
}

