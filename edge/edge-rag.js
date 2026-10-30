/**
 * Edge RAG Implementation using Transformers.js + WebGPU
 *
 * Features:
 * - In-browser vector search with ONNX embeddings
 * - Local SLM for text generation
 * - Dynamic fallback to cloud API for complex queries
 * - Offline-first with cached models
 *
 * Usage:
 *   import { EdgeRAG } from './edge-rag.js';
 *   const rag = new EdgeRAG({ cloudEndpoint: 'https://api.example.com/rag' });
 *   await rag.initialize();
 *   const result = await rag.query('What is RAG?', context);
 */

import { pipeline, env, AutoTokenizer, AutoModel } from '@xenova/transformers';
import { VectorDB } from 'vectordb';

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_CONFIG = {
  embeddingModel: 'nomic-ai/nomic-embed-text-v1',
  slmModel: 'Xenova/TinyLlama-1.1B-Chat-v1.0',
  maxEmbeddingLength: 512,
  maxSLMLength: 2048,
  cloudEndpoint: null,
  cloudTimeout: 5000,
  cloudRetryCount: 2,
  enableWebGPU: true,
  cacheDir: './models',
  vectorDim: 768,
};

// =============================================================================
// EdgeRAG Class
// =============================================================================

export class EdgeRAG {
  constructor(config = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.embeddingModel = null;
    this.slmModel = null;
    this.vectorDB = null;
    this.isInitialized = false;
    this.cloudAvailable = false;
  }

  async initialize() {
    console.log('Initializing Edge RAG...');

    // Check WebGPU support
    if (this.config.enableWebGPU && navigator.gpu) {
      console.log('WebGPU available, using GPU acceleration');
      env.backends.onnx.wasm.proxy = false;
    } else {
      console.log('WebGPU not available, using WASM backend');
    }

    // Initialize embedding model
    console.log('Loading embedding model...');
    this.embeddingModel = await pipeline(
      'feature-extraction',
      this.config.embeddingModel,
      {
        quantized: true,
        cache_dir: this.config.cacheDir,
      }
    );

    // Initialize SLM
    console.log('Loading SLM...');
    this.slmModel = await pipeline(
      'text-generation',
      this.config.slmModel,
      {
        quantized: true,
        cache_dir: this.config.cacheDir,
      }
    );

    // Initialize vector database
    console.log('Initializing vector database...');
    this.vectorDB = new VectorDB({
      dim: this.config.vectorDim,
      metric: 'cosine',
    });

    // Test cloud connectivity
    if (this.config.cloudEndpoint) {
      await this.testCloudConnection();
    }

    this.isInitialized = true;
    console.log('Edge RAG initialized successfully');
  }

  async testCloudConnection() {
    try {
      const response = await fetch(this.config.cloudEndpoint, {
        method: 'GET',
        signal: AbortSignal.timeout(this.config.cloudTimeout),
      });
      this.cloudAvailable = response.ok;
      console.log(`Cloud endpoint ${this.cloudAvailable ? 'available' : 'unavailable'}`);
    } catch (error) {
      this.cloudAvailable = false;
      console.log('Cloud endpoint unavailable, using offline mode');
    }
  }

  async embed(text) {
    if (!this.isInitialized) {
      throw new Error('EdgeRAG not initialized');
    }

    const output = await this.embeddingModel(text, {
      pooling: 'cls',
      normalize: true,
    });

    return Array.from(output.data);
  }

  async embedBatch(texts) {
    const embeddings = await Promise.all(
      texts.map((text) => this.embed(text))
    );
    return embeddings;
  }

  async addDocuments(documents) {
    const texts = documents.map((doc) => doc.text);
    const embeddings = await this.embedBatch(texts);

    const points = documents.map((doc, i) => ({
      id: doc.id || `doc_${Date.now()}_${i}`,
      vector: embeddings[i],
      metadata: {
        text: doc.text,
        ...doc.metadata,
      },
    }));

    await this.vectorDB.upsert(points);
    console.log(`Added ${documents.length} documents to vector DB`);
  }

  async search(query, topK = 5) {
    const queryEmbedding = await this.embed(query);
    const results = await this.vectorDB.search(queryEmbedding, topK);
    return results;
  }

  async generateLocal(query, context, chatHistory = []) {
    const systemPrompt =
      'You are a helpful RAG assistant. Answer based ONLY on the provided context. ' +
      'If the context doesn\'t contain enough information, say so.';

    const contextText = context
      .map((doc) => doc.metadata?.text || doc.text)
      .join('\n\n');

    const messages = [
      { role: 'system', content: systemPrompt },
      ...chatHistory,
      {
        role: 'user',
        content: `Context:\n${contextText}\n\nQuestion: ${query}`,
      },
    ];

    const output = await this.slmModel(messages, {
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.9,
    });

    return output[0].generated_text;
  }

  async generateCloud(query, context, chatHistory = []) {
    if (!this.cloudAvailable || !this.config.cloudEndpoint) {
      throw new Error('Cloud endpoint not available');
    }

    const contextText = context
      .map((doc) => doc.metadata?.text || doc.text)
      .join('\n\n');

    const response = await fetch(this.config.cloudEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        context: contextText,
        chat_history: chatHistory,
      }),
      signal: AbortSignal.timeout(this.config.cloudTimeout),
    });

    if (!response.ok) {
      throw new Error(`Cloud API error: ${response.status}`);
    }

    return await response.json();
  }

  shouldUseCloud(query, context) {
    // Use cloud for complex queries
    const complexKeywords = [
      'search', 'web', 'current', 'latest', 'news', 'today',
      'calculate', 'compute', 'analyze', 'compare',
    ];

    const hasComplexKeyword = complexKeywords.some((kw) =>
      query.toLowerCase().includes(kw)
    );

    // Use cloud if context is insufficient
    const hasInsufficientContext = context.length === 0;

    // Use cloud for long queries
    const isLongQuery = query.length > 200;

    return (
      this.cloudAvailable &&
      (hasComplexKeyword || hasInsufficientContext || isLongQuery)
    );
  }

  async query(question, chatHistory = []) {
    if (!this.isInitialized) {
      throw new Error('EdgeRAG not initialized');
    }

    console.log(`Processing query: "${question.substring(0, 50)}..."`);

    // Search for relevant context
    const searchResults = await this.search(question, 5);
    console.log(`Found ${searchResults.length} relevant documents`);

    // Decide whether to use cloud or local
    const useCloud = this.shouldUseCloud(question, searchResults);

    if (useCloud) {
      console.log('Using cloud API for complex query');
      try {
        const cloudResult = await this.generateCloud(
          question,
          searchResults,
          chatHistory
        );
        return {
          answer: cloudResult.generation,
          source: 'cloud',
          context: searchResults,
          usage: cloudResult.usage,
        };
      } catch (error) {
        console.warn('Cloud API failed, falling back to local:', error);
      }
    }

    // Use local SLM
    console.log('Using local SLM');
    const localAnswer = await this.generateLocal(question, searchResults, chatHistory);
    return {
      answer: localAnswer,
      source: 'local',
      context: searchResults,
    };
  }

  async clearDatabase() {
    await this.vectorDB.deleteAll();
    console.log('Vector database cleared');
  }

  getStats() {
    return {
      initialized: this.isInitialized,
      cloudAvailable: this.cloudAvailable,
      documentCount: this.vectorDB?.count() || 0,
      embeddingModel: this.config.embeddingModel,
      slmModel: this.config.slmModel,
      webgpuEnabled: this.config.enableWebGPU && !!navigator.gpu,
    };
  }
}

// =============================================================================
// Browser Integration
// =============================================================================

export class EdgeRAGUI {
  constructor(containerId, config = {}) {
    this.container = document.getElementById(containerId);
    this.rag = new EdgeRAG(config);
    this.chatHistory = [];
    this.isProcessing = false;
  }

  async initialize() {
    this.renderLoadingUI();
    await this.rag.initialize();
    this.renderChatUI();
    this.updateStats();
  }

  renderLoadingUI() {
    this.container.innerHTML = `
      <div class="edge-rag-loading">
        <div class="spinner"></div>
        <p>Loading AI models...</p>
        <p class="subtitle">This may take a moment on first load</p>
      </div>
    `;
  }

  renderChatUI() {
    this.container.innerHTML = `
      <div class="edge-rag-container">
        <div class="edge-rag-header">
          <h2>Edge RAG Assistant</h2>
          <div class="stats" id="rag-stats"></div>
        </div>
        <div class="chat-messages" id="chat-messages"></div>
        <div class="chat-input">
          <textarea
            id="user-input"
            placeholder="Ask a question..."
            rows="3"
          ></textarea>
          <button id="send-btn" onclick="window.edgeRAGUI.sendMessage()">
            Send
          </button>
        </div>
      </div>
    `;

    // Store reference for onclick handler
    window.edgeRAGUI = this;

    // Add enter key listener
    document.getElementById('user-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
  }

  updateStats() {
    const stats = this.rag.getStats();
    const statsEl = document.getElementById('rag-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <span class="stat ${stats.cloudAvailable ? 'online' : 'offline'}">
          ${stats.cloudAvailable ? '☁️ Cloud' : '📱 Offline'}
        </span>
        <span class="stat">${stats.documentCount} docs</span>
        <span class="stat">${stats.webgpuEnabled ? '🎮 GPU' : '💻 CPU'}</span>
      `;
    }
  }

  async sendMessage() {
    if (this.isProcessing) return;

    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    this.isProcessing = true;
    input.value = '';

    // Add user message to UI
    this.addMessage('user', message);

    try {
      // Get RAG response
      const result = await this.rag.query(message, this.chatHistory);

      // Add to chat history
      this.chatHistory.push(
        { role: 'user', content: message },
        { role: 'assistant', content: result.answer }
      );

      // Add assistant message to UI
      this.addMessage('assistant', result.answer, {
        source: result.source,
        docCount: result.context?.length || 0,
      });
    } catch (error) {
      this.addMessage('error', `Error: ${error.message}`);
    } finally {
      this.isProcessing = false;
      this.updateStats();
    }
  }

  addMessage(role, content, metadata = {}) {
    const messagesEl = document.getElementById('chat-messages');
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;

    let metaHtml = '';
    if (metadata.source) {
      metaHtml = `
        <div class="metadata">
          <span class="source">${metadata.source === 'cloud' ? '☁️ Cloud' : '📱 Local'}</span>
          <span class="docs">${metadata.docCount} documents used</span>
        </div>
      `;
    }

    messageEl.innerHTML = `
      <div class="content">${this.escapeHtml(content)}</div>
      ${metaHtml}
    `;

    messagesEl.appendChild(messageEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// =============================================================================
// Tauri/Electron Integration
// =============================================================================

export class DesktopRAG extends EdgeRAG {
  constructor(config = {}) {
    super({
      ...config,
      enableWebGPU: true,
      cacheDir: await getDesktopCacheDir(),
    });
  }

  async syncWithCloud() {
    // Sync vector DB with cloud for offline access
    if (!this.cloudAvailable) return;

    try {
      const response = await fetch(`${this.config.cloudEndpoint}/documents`, {
        method: 'GET',
      });
      const documents = await response.json();

      await this.addDocuments(documents);
      console.log(`Synced ${documents.length} documents from cloud`);
    } catch (error) {
      console.error('Cloud sync failed:', error);
    }
  }

  async exportForOffline(outputPath) {
    // Export models and vector DB for offline use
    const exportData = {
      config: this.config,
      vectorDB: await this.vectorDB.export(),
      timestamp: new Date().toISOString(),
    };

    // Use Tauri/Electron file system API
    if (window.__TAURI__) {
      await window.__TAURI__.fs.writeTextFile(
        outputPath,
        JSON.stringify(exportData)
      );
    } else if (window.require) {
      const fs = window.require('fs');
      fs.writeFileSync(outputPath, JSON.stringify(exportData));
    }
  }
}

async function getDesktopCacheDir() {
  if (window.__TAURI__) {
    const { appDataDir } = await window.__TAURI__.path;
    return await appDataDir();
  } else if (window.require) {
    const { app } = window.require('electron');
    return app.getPath('userData');
  }
  return './models';
}

export default EdgeRAG;
