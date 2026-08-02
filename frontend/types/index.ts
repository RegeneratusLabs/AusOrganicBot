export interface SourceDocument {
  content: string;
  metadata: {
    source?: string;
    filename?: string;
    page_number?: number;
    chunk_index?: number;
    [key: string]: any;
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
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocument[];
  reasoning?: string;
  timestamp: Date;
}

export interface SystemStatus {
  documents_loaded: number;
  llm_connected: boolean;
  model: string;
}
