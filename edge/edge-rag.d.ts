/**
 * Edge RAG TypeScript Definitions
 *
 * Type definitions for the Edge RAG implementation.
 */

import { pipeline, env } from '@xenova/transformers';

// =============================================================================
// Types
// =============================================================================

export interface EdgeRAGConfig {
  embeddingModel?: string;
  slmModel?: string;
  maxEmbeddingLength?: number;
  maxSLMLength?: number;
  cloudEndpoint?: string | null;
  cloudTimeout?: number;
  cloudRetryCount?: number;
  enableWebGPU?: boolean;
  cacheDir?: string;
  vectorDim?: number;
}

export interface Document {
  id?: string;
  text: string;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  id: string;
  vector: number[];
  metadata: {
    text: string;
    [key: string]: any;
  };
  score?: number;
}

export interface RAGResponse {
  answer: string;
  source: 'local' | 'cloud';
  context: SearchResult[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface RAGStats {
  initialized: boolean;
  cloudAvailable: boolean;
  documentCount: number;
  embeddingModel: string;
  slmModel: string;
  webgpuEnabled: boolean;
}

// =============================================================================
// EdgeRAG Class
// =============================================================================

export declare class EdgeRAG {
  constructor(config?: EdgeRAGConfig);

  initialize(): Promise<void>;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  addDocuments(documents: Document[]): Promise<void>;
  search(query: string, topK?: number): Promise<SearchResult[]>;
  query(question: string, chatHistory?: Array<{ role: string; content: string }>): Promise<RAGResponse>;
  clearDatabase(): Promise<void>;
  getStats(): RAGStats;
}

// =============================================================================
// EdgeRAGUI Class
// =============================================================================

export declare class EdgeRAGUI {
  constructor(containerId: string, config?: EdgeRAGConfig);

  initialize(): Promise<void>;
  sendMessage(): Promise<void>;
}

// =============================================================================
// DesktopRAG Class
// =============================================================================

export declare class DesktopRAG extends EdgeRAG {
  constructor(config?: EdgeRAGConfig);

  syncWithCloud(): Promise<void>;
  exportForOffline(outputPath: string): Promise<void>;
}

// =============================================================================
// Utility Types
// =============================================================================

export interface VectorDBConfig {
  dim: number;
  metric: 'cosine' | 'euclidean' | 'dot';
}

export interface CloudEndpointResponse {
  generation: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// =============================================================================
// Environment Detection
// =============================================================================

export interface Environment {
  isBrowser: boolean;
  isNode: boolean;
  isTauri: boolean;
  isElectron: boolean;
  hasWebGPU: boolean;
}

export declare function detectEnvironment(): Environment;
