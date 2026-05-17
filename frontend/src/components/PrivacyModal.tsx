import { useCallback, useEffect, useState } from "react";
import { ApiError, setConsent } from "../lib/api";
import type { ConsentTier } from "../types";

// Bumping ``PRIVACY_NOTICE_VERSION`` invalidates prior acknowledgements,
// so users see the modal again the next time material copy changes —
// the standard versioning pattern used by cookie / consent banners.
const PRIVACY_NOTICE_VERSION = "1";
export const CONSENT_STORAGE_KEY = `multiagent_graphrag_consent_v${PRIVACY_NOTICE_VERSION}`;
export const CONSENT_SEEN_KEY = `${CONSENT_STORAGE_KEY}__ack`;

function markSeen(): void {
  try {
    localStorage.setItem(CONSENT_SEEN_KEY, "1");
  } catch {
    // Persistence is best-effort; if localStorage is unavailable the modal
    // will simply re-appear on the next visit.
  }
}

interface PrivacyModalProps {
  open: boolean;
  onClose: () => void;
}

interface TierOption {
  value: ConsentTier;
  label: string;
  blurb: string;
}

const OPTIONS: TierOption[] = [
  {
    value: "full",
    label: "Full",
    blurb: "Share full queries and answers to help improve the app.",
  },
  {
    value: "anonymised",
    label: "Anonymised",
    blurb:
      "Share queries and answers with personal details (emails, phone numbers, identifiers) masked before they leave your machine.",
  },
  {
    value: "metadata_only",
    label: "Metadata only",
    blurb:
      "Share only timings and run structure. Your inputs and outputs are never recorded.",
  },
];

export function PrivacyModal({ open, onClose }: PrivacyModalProps) {
  const [selected, setSelected] = useState<ConsentTier | null>(null);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const handleDismiss = useCallback(() => {
    markSeen();
    onClose();
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    try {
      const stored = localStorage.getItem(CONSENT_STORAGE_KEY);
      if (stored === "full" || stored === "anonymised" || stored === "metadata_only") {
        setSelected(stored);
      } else {
        setSelected(null);
      }
    } catch {
      setSelected(null);
    }
    setError("");
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") handleDismiss();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, handleDismiss]);

  // Lock body scroll while open so the backdrop click target is unambiguous
  // and the page underneath can't be scrolled by accident.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open) return null;

  async function handleSave() {
    if (selected === null) return;
    setBusy(true);
    setError("");
    try {
      await setConsent(selected);
      try {
        localStorage.setItem(CONSENT_STORAGE_KEY, selected);
      } catch {
        // Persistence is best-effort; the choice is still recorded server-side
        // for this session.
      }
      markSeen();
      onClose();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err instanceof Error ? err.message : "Failed to save choice");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-labelledby="privacy-modal-title"
      onClick={(event) => {
        if (event.target === event.currentTarget) handleDismiss();
      }}
    >
      <div className="modal-panel">
        <button
          type="button"
          className="modal-close"
          onClick={handleDismiss}
          aria-label="Close privacy preview"
          disabled={busy}
        >
          <span aria-hidden="true">×</span>
        </button>
        <header className="modal-header">
          <h2 id="privacy-modal-title">
            <span className="title-mark">◆</span> Privacy preview
            <span className="modal-badge" title="Demo only: has no effect in clone-and-run">
              demo · inert in local run
            </span>
          </h2>
        </header>

        <div className="modal-body">
          <p className="modal-lede">
            This dialog is a <strong>demo</strong> of the data-sharing choice
            that would govern run tracing if this app were hosted by the
            author. Right now the app runs entirely on your machine, no data
            leaves it, and your choice below has no effect — it&apos;s
            preserved for when a hosted version goes live.
          </p>
          <p className="modal-lede modal-lede-soft">
            You can reopen this any time from <strong>Privacy preferences</strong> in the sidebar.
          </p>

          <p className="modal-section-label">
            When tracing is live, you would choose:
          </p>

          <fieldset className="consent-options">
            <legend className="visually-hidden">
              Data-sharing tier (preview only)
            </legend>
            {OPTIONS.map((option) => {
              const id = `consent-${option.value}`;
              const checked = selected === option.value;
              return (
                <label
                  key={option.value}
                  htmlFor={id}
                  className={`consent-option ${
                    checked ? "is-selected" : ""
                  }`}
                >
                  <input
                    id={id}
                    type="radio"
                    name="consent-tier"
                    value={option.value}
                    checked={checked}
                    onChange={() => setSelected(option.value)}
                    disabled={busy}
                  />
                  <span className="consent-option-text">
                    <span className="consent-option-label">{option.label}</span>
                    <span className="consent-option-blurb">{option.blurb}</span>
                  </span>
                </label>
              );
            })}
          </fieldset>

          {error !== "" && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
        </div>

        <footer className="modal-footer">
          <button
            type="button"
            className="ghost"
            onClick={handleDismiss}
            disabled={busy}
          >
            Dismiss
          </button>
          <button
            type="button"
            className="primary"
            onClick={handleSave}
            disabled={busy || selected === null}
          >
            {busy ? "Saving…" : "Save choice"}
          </button>
        </footer>
      </div>
    </div>
  );
}
