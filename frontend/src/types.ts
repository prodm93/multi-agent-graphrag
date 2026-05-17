export interface AuraCreds {
  neo4jUri: string;
  neo4jUsername: string;
  neo4jPassword: string;
  neo4jDatabase: string;
  auraInstanceId: string;
  auraInstanceName: string;
}

export interface CredentialsPayload extends AuraCreds {
  openaiKey: string;
}

export interface IngestResponse {
  message: string;
  files_received: number;
  successes: number;
  failures: number;
  failed_files: string[];
}

export interface QueryResponse {
  answer: string;
}

export type ConsentTier = "full" | "anonymised" | "metadata_only";
