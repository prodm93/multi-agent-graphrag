import { useState } from "react";
import { CredentialsProvider } from "./context/CredentialsContext";
import { Sidebar } from "./components/Sidebar";
import { DocumentUpload } from "./components/DocumentUpload";
import { QueryInterface } from "./components/QueryInterface";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);

  return (
    <CredentialsProvider>
      <div className={`app-shell ${sidebarOpen ? "is-open" : "is-collapsed"}`}>
        <Sidebar
          open={sidebarOpen}
          onToggle={() => setSidebarOpen((prev) => !prev)}
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
    </CredentialsProvider>
  );
}
