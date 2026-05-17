import { useEffect, useState } from "react";
import { CredentialsProvider } from "./context/CredentialsContext";
import { Sidebar } from "./components/Sidebar";
import { DocumentUpload } from "./components/DocumentUpload";
import { QueryInterface } from "./components/QueryInterface";
import { CONSENT_SEEN_KEY, PrivacyModal } from "./components/PrivacyModal";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [privacyOpen, setPrivacyOpen] = useState<boolean>(false);

  useEffect(() => {
    // Soft modal: open until the user explicitly acknowledges it (dismiss
    // or save). The "seen" flag is written by the modal itself on those
    // actions — not on first show — so a refresh before any interaction
    // still re-opens the modal. After acknowledgement, the only way to
    // re-open it is the Privacy link in the sidebar.
    try {
      const seen = localStorage.getItem(CONSENT_SEEN_KEY);
      if (seen !== "1") {
        setPrivacyOpen(true);
      }
    } catch {
      // localStorage unavailable — fall back to showing the modal once
      // per session (no persistence).
      setPrivacyOpen(true);
    }
  }, []);

  return (
    <CredentialsProvider>
      <div className={`app-shell ${sidebarOpen ? "is-open" : "is-collapsed"}`}>
        <Sidebar
          open={sidebarOpen}
          onToggle={() => setSidebarOpen((prev) => !prev)}
          onOpenPrivacy={() => setPrivacyOpen(true)}
        />
        <main className="app-main">
          <header className="app-header">
            <button
              type="button"
              className="sidebar-toggle"
              onClick={() => setSidebarOpen((prev) => !prev)}
              aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
              aria-expanded={sidebarOpen}
            >
              <span aria-hidden="true">{sidebarOpen ? "‹" : "›"}</span>
            </button>
            <h1>
              <span className="title-mark">◆</span> Multi-agent GraphRAG
            </h1>
          </header>
          <DocumentUpload />
          <QueryInterface />
        </main>
      </div>
      <PrivacyModal
        open={privacyOpen}
        onClose={() => setPrivacyOpen(false)}
      />
    </CredentialsProvider>
  );
}
