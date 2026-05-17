import { useState, type ChangeEvent } from "react";
import { useCredentials } from "../context/CredentialsContext";
import { ApiError, parseAuraCredentials, setCredentials } from "../lib/api";
import type { CredentialsPayload } from "../types";

interface SidebarProps {
  open: boolean;
  onToggle: () => void;
  onOpenPrivacy: () => void;
}

const EMPTY_FORM: CredentialsPayload = {
  neo4jUri: "",
  neo4jUsername: "",
  neo4jPassword: "",
  neo4jDatabase: "",
  auraInstanceId: "",
  auraInstanceName: "",
  openaiKey: "",
};

export function Sidebar({ open, onToggle, onOpenPrivacy }: SidebarProps) {
  const { isReady, setReady } = useCredentials();
  const [form, setForm] = useState<CredentialsPayload>(EMPTY_FORM);
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);
  const [parsing, setParsing] = useState<boolean>(false);

  async function handleAuraFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setParsing(true);
    try {
      const parsed = await parseAuraCredentials(file);
      setForm((prev) => ({ ...prev, ...parsed }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse credentials file");
    } finally {
      setParsing(false);
    }
  }

  function handleFieldChange(field: keyof CredentialsPayload, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function handleConnect() {
    setError("");
    if (!form.openaiKey.trim()) {
      setError("OpenAI API key is required");
      return;
    }
    const hasUri = form.neo4jUri.trim() !== "";
    const hasAuraPair =
      form.auraInstanceId.trim() !== "" && form.auraInstanceName.trim() !== "";
    if (!hasUri && !hasAuraPair) {
      setError("Provide a Neo4j URI, or both AURA_INSTANCEID and AURA_INSTANCENAME");
      return;
    }
    if (
      !form.neo4jUsername.trim() ||
      !form.neo4jPassword ||
      !form.neo4jDatabase.trim()
    ) {
      setError("Neo4j username, password, and database are required");
      return;
    }

    setBusy(true);
    try {
      await setCredentials(form);
      setReady(true);
    } catch (err) {
      setReady(false);
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err instanceof Error ? err.message : "Failed to set credentials");
      }
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <aside className="sidebar sidebar-collapsed" aria-hidden="false">
        <button
          type="button"
          className="sidebar-toggle sidebar-toggle-rail"
          onClick={onToggle}
          aria-label="Expand sidebar"
          aria-expanded={false}
        >
          <span aria-hidden="true">›</span>
        </button>
        <div className="sidebar-rail-status" aria-live="polite">
          <span
            className={`status-dot ${isReady ? "is-ready" : "is-idle"}`}
            aria-hidden="true"
          />
        </div>
      </aside>
    );
  }

  return (
    <aside className="sidebar sidebar-open">
      <div className="sidebar-header">
        <h2>Credentials</h2>
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggle}
          aria-label="Collapse sidebar"
          aria-expanded={true}
        >
          <span aria-hidden="true">‹</span>
        </button>
      </div>

      <section className="sidebar-section">
        <label htmlFor="aura-file">Neo4j credentials .txt</label>
        <input
          id="aura-file"
          type="file"
          accept=".txt"
          onChange={handleAuraFile}
          disabled={parsing || busy}
          aria-describedby={error !== "" ? "creds-error" : undefined}
        />
        <p className="hint">
          Upload the Aura .txt to auto-fill the fields below, or fill them in
          manually.
        </p>
      </section>

      <section className="sidebar-section">
        <label htmlFor="neo4j-uri">Neo4j URI</label>
        <input
          id="neo4j-uri"
          type="text"
          autoComplete="off"
          value={form.neo4jUri}
          onChange={(e) => handleFieldChange("neo4jUri", e.target.value)}
          placeholder="neo4j+s://xxxxxx.databases.neo4j.io"
        />

        <label htmlFor="neo4j-username">Neo4j username</label>
        <input
          id="neo4j-username"
          type="text"
          autoComplete="off"
          value={form.neo4jUsername}
          onChange={(e) => handleFieldChange("neo4jUsername", e.target.value)}
        />

        <label htmlFor="neo4j-password">Neo4j password</label>
        <input
          id="neo4j-password"
          type="password"
          autoComplete="off"
          value={form.neo4jPassword}
          onChange={(e) => handleFieldChange("neo4jPassword", e.target.value)}
        />

        <label htmlFor="neo4j-database">Neo4j database</label>
        <input
          id="neo4j-database"
          type="text"
          autoComplete="off"
          value={form.neo4jDatabase}
          onChange={(e) => handleFieldChange("neo4jDatabase", e.target.value)}
        />
      </section>

      <section className="sidebar-section">
        <h3 className="section-subtitle">AuraDB (optional)</h3>
        <p className="hint">
          If both are set and Neo4j URI is empty, the URI is derived as
          <code> neo4j+s://&lt;id&gt;.databases.neo4j.io</code>.
        </p>

        <label htmlFor="aura-instance-id">AURA_INSTANCEID</label>
        <input
          id="aura-instance-id"
          type="text"
          autoComplete="off"
          value={form.auraInstanceId}
          onChange={(e) => handleFieldChange("auraInstanceId", e.target.value)}
          placeholder="e.g. abc12345"
        />

        <label htmlFor="aura-instance-name">AURA_INSTANCENAME</label>
        <input
          id="aura-instance-name"
          type="text"
          autoComplete="off"
          value={form.auraInstanceName}
          onChange={(e) => handleFieldChange("auraInstanceName", e.target.value)}
          placeholder="e.g. Instance01"
        />
      </section>

      <section className="sidebar-section">
        <label htmlFor="openai-key">OpenAI API key</label>
        <input
          id="openai-key"
          type="password"
          autoComplete="off"
          aria-label="OpenAI API key"
          value={form.openaiKey}
          onChange={(e) => handleFieldChange("openaiKey", e.target.value)}
        />
      </section>

      <button
        type="button"
        className="primary"
        onClick={handleConnect}
        disabled={busy || parsing}
      >
        {busy ? "Connecting…" : isReady ? "Reconnect" : "Connect"}
      </button>

      {error !== "" && (
        <p id="creds-error" role="alert" className="error">
          {error}
        </p>
      )}
      <div className="sidebar-status" aria-live="polite">
        <span
          className={`status-dot ${isReady ? "is-ready" : "is-idle"}`}
          aria-hidden="true"
        />
        <span>{isReady ? "Connected" : "Not connected"}</span>
      </div>

      <div className="sidebar-footer">
        <button
          type="button"
          className="sidebar-privacy"
          onClick={onOpenPrivacy}
        >
          <span className="sidebar-privacy-icon" aria-hidden="true">◆</span>
          <span className="sidebar-privacy-label">Privacy preferences</span>
          <span className="sidebar-privacy-chevron" aria-hidden="true">›</span>
        </button>
      </div>
    </aside>
  );
}
