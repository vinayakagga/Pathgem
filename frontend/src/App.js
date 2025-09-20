import React, { useState, useEffect } from 'react';
import { Scale, Book, FileText, Search, Shield, Send } from 'lucide-react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState({});

  // API base URL - FIXED: using absolute URL
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    fetchDocuments();
    fetchStats();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE}/documents/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setDocuments([]);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setResponse(null);
    
    try {
      const response = await fetch(
        `${API_BASE}/query/?question=${encodeURIComponent(question)}&top_k=${topK}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResponse(data);
    } catch (error) {
      setResponse({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const presetQuestions = [
    "Explain Article 21 of the Indian Constitution",
    "What are the fundamental rights under Indian law?",
    "Describe the process of criminal trial in India",
    "What is the difference between IPC and CrPC?",
    "Explain the concept of habeas corpus",
    "What are the grounds for divorce under Hindu Marriage Act?"
  ];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Scale size={32} />
            <div>
              <h1>Pathgem</h1>
              <p>AI-Powered Indian Legal Research Assistant</p>
            </div>
          </div>
          <div className="header-stats">
            <span>🇮🇳 Indian Law Focused</span>
            <span>🔒 Secure & Private</span>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <h3><FileText size={20} /> System Status</h3>
            {stats.ollama_connected && (
              <div className="status-item">
                <span className="status-online">●</span>
                <span>AI Model: {stats.ollama_model}</span>
              </div>
            )}
            {stats.document_count > 0 && (
              <div className="status-item">
                <span>📄 Documents: {stats.document_count}</span>
              </div>
            )}
            {stats.ollama_host && (
              <div className="status-item">
                <span>🌐 Host: {stats.ollama_host}</span>
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <h3><Book size={20} /> Legal Documents</h3>
            <div className="document-list">
              {documents.length > 0 ? (
                documents.slice(0, 5).map((doc, index) => (
                  <div key={index} className="document-item">
                    <div className="document-name">
                      {doc.path.split('/').pop()}
                    </div>
                    <div className="document-preview">
                      {doc.preview}
                    </div>
                  </div>
                ))
              ) : (
                <p>No documents loaded</p>
              )}
            </div>
            <button onClick={fetchDocuments} className="refresh-btn">
              Refresh Documents
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <div className="query-section">
            <h2><Search size={24} /> Legal Research</h2>
            
            {/* Preset Questions */}
            <div className="preset-questions">
              <h4>Common Legal Queries:</h4>
              <div className="preset-grid">
                {presetQuestions.map((q, index) => (
                  <button
                    key={index}
                    className="preset-btn"
                    onClick={() => setQuestion(q)}
                    disabled={loading}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Query Input */}
            <div className="query-input">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask about Indian laws, constitutional articles, legal procedures..."
                rows={4}
                disabled={loading}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                  }
                }}
              />
              
              <div className="query-controls">
                <div className="control-group">
                  <label>Documents to reference:</label>
                  <select 
                    value={topK} 
                    onChange={(e) => setTopK(e.target.value)}
                    disabled={loading}
                  >
                    {[1, 2, 3, 4, 5].map(num => (
                      <option key={num} value={num}>{num}</option>
                    ))}
                  </select>
                </div>
                
                <button 
                  onClick={askQuestion} 
                  disabled={loading || !question.trim()}
                  className="ask-btn"
                >
                  {loading ? (
                    <div className="loading-spinner"></div>
                  ) : (
                    <>
                      <Send size={16} />
                      Ask Question
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Response */}
            {response && (
              <div className="response-section">
                <h3>Legal Analysis:</h3>
                <div className="response-box">
                  {response.error ? (
                    <div className="error">
                      <strong>Error:</strong> {response.error}
                    </div>
                  ) : (
                    <>
                      <div className="response-header">
                        <strong>Question:</strong> {response.question}
                      </div>
                      <div className="response-content">
                        {response.response}
                      </div>
                      <div className="response-footer">
                        <span>📊 Used {response.context_documents_used} legal documents</span>
                        <span>⚖️ Model: {response.model_used}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Researching legal documents...</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;