import type {
  AuraCreds,
  CredentialsPayload,
  IngestResponse,
  QueryResponse,
} from "../types";

interface ParsedCredentialsResponse {
  neo4j_uri: string;
  neo4j_username: string;
  neo4j_password: string;
  neo4j_database: string;
  aura_instanceid?: string;
  aura_instancename?: string;
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function extractErrorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.clone().json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.length > 0) {
      return body.detail;
    }
  } catch {
    // Fall through to status text.
  }
  return res.statusText || `HTTP ${res.status}`;
}

const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

class ApiClient {
  constructor(private readonly baseUrl: string = "/api") {}

  private async exec<T>(path: string, init: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new ApiError(408, "Request timed out");
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      throw new ApiError(res.status, await extractErrorDetail(res));
    }
    return (await res.json()) as T;
  }

  async postJson<T>(path: string, body: unknown): Promise<T> {
    return this.exec<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  async postMultipart<T>(path: string, form: FormData): Promise<T> {
    return this.exec<T>(path, {
      method: "POST",
      body: form,
    });
  }
}

export const apiClient = new ApiClient();
export { ApiClient, ApiError };

export async function ingest(files: File[]): Promise<IngestResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  return apiClient.postMultipart<IngestResponse>("/ingest", form);
}

export async function query(question: string): Promise<QueryResponse> {
  return apiClient.postJson<QueryResponse>("/query", { query: question });
}

export async function setCredentials(
  payload: CredentialsPayload,
): Promise<void> {
  await apiClient.postJson<{ status: string }>("/credentials", {
    neo4j_uri: payload.neo4jUri,
    neo4j_username: payload.neo4jUsername,
    neo4j_password: payload.neo4jPassword,
    neo4j_database: payload.neo4jDatabase,
    openai_api_key: payload.openaiKey,
    aura_instanceid: payload.auraInstanceId,
    aura_instancename: payload.auraInstanceName,
  });
}

export async function parseAuraCredentials(file: File): Promise<AuraCreds> {
  const form = new FormData();
  form.append("file", file);
  const parsed = await apiClient.postMultipart<ParsedCredentialsResponse>(
    "/credentials/parse",
    form,
  );
  return {
    neo4jUri: parsed.neo4j_uri,
    neo4jUsername: parsed.neo4j_username,
    neo4jPassword: parsed.neo4j_password,
    neo4jDatabase: parsed.neo4j_database,
    auraInstanceId: parsed.aura_instanceid ?? "",
    auraInstanceName: parsed.aura_instancename ?? "",
  };
}
