export interface SourceDocument {
  content: string;
  metadata: {
    source?: string;
    filename?: string;
    page_number?: number;
    chunk_index?: number;
    edition?: string;
    country_code?: string;
    [key: string]: unknown;
  };
  score: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  confidence: number;
  model: string;
  processing_time_ms: number;
  reasoning?: string;
  country?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocument[];
  reasoning?: string;
  timestamp: Date;
}

export interface CountryStandard {
  id: string;
  name: string;
  edition: string;
  /** Document licence (e.g. 'CC-BY-4.0', 'all rights reserved') or null. */
  license?: string | null;
  /** Whether full copies (PDF/HTML) may be served by this deployment. */
  redistributable?: boolean;
}

export interface CountryInfo {
  code: string;
  name: string;
  language: string;
  standards: CountryStandard[];
}

export interface SystemStatus {
  documents_loaded: number;
  llm_connected: boolean;
  model: string;
  countries: CountryInfo[];
  default_country: string;
}
