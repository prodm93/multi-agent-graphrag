import { useState } from "react";
import { ApiError, query as runQuery } from "../lib/api";
import { useCredentials } from "../context/CredentialsContext";

export function QueryInterface() {
  const creds = useCredentials();
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState<boolean>(false);

  async function handleSubmit() {
    if (question.trim() === "") {
      setStatus("Enter a question first.");
      return;
    }
    setBusy(true);
    setStatus("");
    setAnswer("");
    try {
      const result = await runQuery(question);
      const text = result.answer?.trim() ?? "";
      if (text === "") {
        setStatus("No answer found in the knowledge graph.");
      } else {
        setAnswer(text);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 408 || err.status === 504) {
          setStatus("Request timed out, try again.");
        } else if (err.status === 501) {
          setStatus(err.message || "Endpoint not yet implemented.");
        } else {
          setStatus(err.message);
        }
      } else {
        setStatus(err instanceof Error ? err.message : "Query failed");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2>Ask</h2>
      <textarea
        rows={4}
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question grounded in the knowledge graph..."
        aria-label="Your question"
        disabled={busy}
      />
      <button
        type="button"
        onClick={handleSubmit}
        disabled={busy || !creds.isReady || question.trim() === ""}
      >
        {busy ? "Querying…" : "Submit"}
      </button>

      {busy && (
        <div
          role="status"
          aria-live="polite"
          aria-label="Loading answer"
          className="skeleton"
        >
          <div className="skeleton-line" />
          <div className="skeleton-line short" />
          <div className="skeleton-line" />
        </div>
      )}

      {!busy && status !== "" && (
        <p role="status" aria-live="polite">
          {status}
        </p>
      )}

      {!busy && answer !== "" && (
        <article aria-live="polite">
          <h3>Answer</h3>
          <p>{answer}</p>
        </article>
      )}
    </section>
  );
}
