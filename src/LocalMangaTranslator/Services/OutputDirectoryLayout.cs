namespace LocalMangaTranslator.Services;

/// <summary>
/// Canonical output folder layout.
/// The output root is intentionally reserved for final translated images.
/// </summary>
public static class OutputDirectoryLayout
{
    public const string DebugFolderName = "debug";
    public const string JsonFolderName = "json";
    public const string AuditFolderName = "완료검토로그";

    public static string Debug(string root)
        => Path.Combine(root, DebugFolderName);

    public static string Json(string root)
        => Path.Combine(root, JsonFolderName);

    public static string Audit(string root)
        => Path.Combine(root, AuditFolderName);

    public static void Ensure(string root)
    {
        Directory.CreateDirectory(root);
        Directory.CreateDirectory(Debug(root));
        Directory.CreateDirectory(Json(root));
        Directory.CreateDirectory(Audit(root));
    }
}
