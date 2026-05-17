import {
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import { ApiError, ingest } from "../lib/api";
import { useCredentials } from "../context/CredentialsContext";

const ACCEPT = ".pdf,.csv,.docx,.xlsx,.json,.txt";

const MIME_BY_SUFFIX: Record<string, string> = {
  pdf: "application/pdf",
  csv: "text/csv",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  json: "application/json",
  txt: "text/plain",
};

const ALLOWED_SUFFIXES = new Set(Object.keys(MIME_BY_SUFFIX));
const MAX_FILE_BYTES = 50_000_000;
const MAX_FILES = 50;

interface FileIssue {
  filename: string;
  reason: string;
}

function fileKey(f: File): string {
  return `${f.name}|${f.size}|${f.lastModified}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1_000) return `${bytes} B`;
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(1)} KB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function suffixFor(file: File): string {
  return file.name.split(".").pop()?.toLowerCase() ?? "";
}

export function DocumentUpload() {
  const creds = useCredentials();
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [graphReady, setGraphReady] = useState<boolean>(false);
  const [issues, setIssues] = useState<FileIssue[]>([]);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef<number>(0);

  const totalBytes = useMemo(
    () => files.reduce((sum, f) => sum + f.size, 0),
    [files],
  );

  function mergeFiles(incoming: File[]) {
    if (incoming.length === 0) {
      setIssues([]);
      setStatus("");
      return;
    }
    // Hard-reject anything that doesn't match an allowed extension AND
    // size cap. Bad files NEVER enter the list — no silent acceptance,
    // no deferred validation at submit time.
    const accepted: File[] = [];
    const rejected: FileIssue[] = [];
    for (const f of incoming) {
      const suffix = suffixFor(f);
      if (!ALLOWED_SUFFIXES.has(suffix)) {
        rejected.push({
          filename: f.name,
          reason: `Unsupported file type ".${suffix || "unknown"}"; allowed: ${[...ALLOWED_SUFFIXES].sort().join(", ")}`,
        });
        continue;
      }
      if (f.size > MAX_FILE_BYTES) {
        rejected.push({
          filename: f.name,
          reason: `File exceeds 50 MB (${(f.size / 1_000_000).toFixed(1)} MB)`,
        });
        continue;
      }
      accepted.push(f);
    }
    setFiles((prev) => {
      const seen = new Set(prev.map(fileKey));
      const merged = [...prev];
      for (const f of accepted) {
        const key = fileKey(f);
        if (!seen.has(key)) {
          merged.push(f);
          seen.add(key);
        }
      }
      return merged.slice(0, MAX_FILES);
    });
    setIssues(rejected);
    setStatus(
      rejected.length > 0
        ? `Rejected ${rejected.length} file${rejected.length === 1 ? "" : "s"} — see below.`
        : "",
    );
  }

  function handleFiles(event: ChangeEvent<HTMLInputElement>) {
    const list = event.target.files;
    if (!list || list.length === 0) return;
    mergeFiles(Array.from(list));
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDragEnter(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (busy) return;
    dragDepthRef.current += 1;
    setDragActive(true);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (busy) return;
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setDragActive(false);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = 0;
    setDragActive(false);
    if (busy) return;
    const dropped = Array.from(event.dataTransfer?.files ?? []);
    mergeFiles(dropped);
  }

  function removeFile(key: string) {
    setFiles((prev) => prev.filter((f) => fileKey(f) !== key));
    setIssues([]);
    setStatus("");
  }

  function clearFiles() {
    setFiles([]);
    setIssues([]);
    setStatus("");
  }

  async function handleSubmit() {
    if (files.length === 0) {
      setStatus("Pick at least one file first.");
      return;
    }
    setBusy(true);
    setStatus(`Uploading ${files.length} document(s) and building the knowledge graph…`);
    setIssues([]);
    try {
      const result = await ingest(files);
      setStatus(result.message);
      if (result.failed_files.length > 0) {
        setIssues(
          result.failed_files.map((filename) => ({
            filename,
            reason: "Ingestion failed",
          })),
        );
      }
      if (result.successes > 0) {
        setGraphReady(true);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setStatus(err.message);
        if (err.status === 400) {
          setIssues([{ filename: "(server)", reason: err.message }]);
        }
      } else {
        setStatus(err instanceof Error ? err.message : "Ingest failed");
      }
    } finally {
      setBusy(false);
    }
  }

  const remaining = MAX_FILES - files.length;
  const ingestBlockedReason = !creds.isReady
    ? "Connect your Neo4j and OpenAI credentials in the sidebar first."
    : files.length === 0
    ? "Add at least one file."
    : "";

  return (
    <section>
      <h2>Documents</h2>
      <div className="upload-controls">
        <label
          className={`drop-zone ${dragActive ? "is-drag-active" : ""} ${
            busy || remaining <= 0 ? "is-disabled" : ""
          }`}
          htmlFor="documents-input"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            id="documents-input"
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPT}
            onChange={handleFiles}
            aria-label="Documents to ingest"
            disabled={busy || remaining <= 0}
            className="drop-zone-input"
          />
          <div className="drop-zone-body">
            <span className="drop-zone-icon" aria-hidden="true">
              ⬆
            </span>
            <span className="drop-zone-primary">
              {remaining > 0
                ? "Drop files here or click to browse"
                : `Limit reached (${MAX_FILES} files)`}
            </span>
            <span className="drop-zone-secondary">
              {remaining > 0
                ? `Add up to ${remaining} more · PDF · CSV · DOCX · XLSX · JSON · TXT`
                : "Clear some files to add more"}
            </span>
          </div>
        </label>
        <div className="upload-actions">
          <button
            type="button"
            className="primary"
            onClick={handleSubmit}
            disabled={busy || !creds.isReady || files.length === 0}
            title={ingestBlockedReason || undefined}
          >
            {busy ? "Ingesting…" : `Ingest ${files.length || ""}`.trim()}
          </button>
          {files.length > 0 && (
            <button
              type="button"
              className="ghost"
              onClick={clearFiles}
              disabled={busy}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      <p className="hint">
        Up to {MAX_FILES} files per ingestion · 50 MB each · PDF, CSV, DOCX,
        XLSX, JSON, TXT
      </p>

      {!creds.isReady && (
        <p className="hint" role="note">
          Connect your Neo4j and OpenAI credentials in the sidebar before
          ingesting.
        </p>
      )}

      {files.length > 0 && (
        <ul className="file-list" aria-label="Selected files">
          {files.map((f) => {
            const key = fileKey(f);
            return (
              <li key={key} className="file-row">
                <span className="file-name" title={f.name}>
                  {f.name}
                </span>
                <span className="file-meta">{formatBytes(f.size)}</span>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => removeFile(key)}
                  disabled={busy}
                  aria-label={`Remove ${f.name}`}
                >
                  ×
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {files.length > 0 && (
        <p className="hint">
          {files.length} file{files.length === 1 ? "" : "s"} selected ·
          {" "}
          {formatBytes(totalBytes)} total
        </p>
      )}

      {graphReady && (
        <p role="status" aria-live="polite" className="success">
          ✓ Knowledge graph ready
        </p>
      )}

      {status !== "" && <p aria-live="polite">{status}</p>}

      {issues.length > 0 && (
        <ul role="alert" className="issues">
          {issues.map((issue, idx) => (
            <li key={idx}>
              <strong>{issue.filename}</strong>: {issue.reason}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
