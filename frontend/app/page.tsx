"use client";

import { FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";

type Source = {
  title: string;
  url: string;
};

type AskResponse = {
  answer: string;
  sources: Source[];
};

const EXAMPLE_QUESTIONS = [
  "What does the Bitovi blog say about React performance?",
  "How does Bitovi approach testing frontend applications?",
  "What articles discuss AI-assisted software development?",
];

const API_URL = process.env.NEXT_PUBLIC_FASTAPI_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError("Enter a question first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResponse(null);

    try {
      const result = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      });
      const payload = await result.json();

      if (!result.ok) {
        throw new Error(payload.detail ?? "The request failed.");
      }

      setResponse(payload);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The request failed. Check the API server."
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="query-panel" aria-labelledby="page-title">
        <div className="brand-row">
          <img src="/bitovi-mark.svg" alt="" className="brand-mark" />
          <span>Bitovi Blog Agent</span>
        </div>

        <h1 id="page-title">Ask the Bitovi blog</h1>
        <p className="intro">
          Get concise answers grounded in indexed Bitovi articles.
        </p>

        <form onSubmit={askQuestion} className="question-form">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about a topic, practice, tool, or article..."
            disabled={isLoading}
            rows={5}
          />
          <div className="form-actions">
            <button type="submit" disabled={isLoading}>
              {isLoading ? "Asking..." : "Ask"}
            </button>
          </div>
        </form>

        <div className="examples" aria-label="Example questions">
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setQuestion(example)}
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      <section className="answer-area" aria-live="polite">
        {isLoading && (
          <div className="loading-state" role="status">
            <span />
            <span />
            <span />
          </div>
        )}

        {error && <p className="error-message">{error}</p>}

        {response && (
          <>
            <article className="answer-panel">
              <h2>Answer</h2>
              <ReactMarkdown>{response.answer}</ReactMarkdown>
            </article>

            <div className="sources">
              <h2>Sources</h2>
              <div className="source-grid">
                {response.sources.map((source) => (
                  <a
                    key={source.url}
                    className="source-card"
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span>{source.title}</span>
                    <small>{source.url}</small>
                  </a>
                ))}
              </div>
            </div>
          </>
        )}

        {!isLoading && !error && !response && (
          <div className="empty-state">
            <h2>Ready</h2>
            <p>Ask a question after the API server and index are ready.</p>
          </div>
        )}
      </section>
    </main>
  );
}
