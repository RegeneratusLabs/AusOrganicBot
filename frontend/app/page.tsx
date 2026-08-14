'use client';

import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { ChatMessage, SystemStatus } from '@/types';
import { formatSourceName } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [country, setCountry] = useState<string>('au');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/status`)
      .then((r) => r.json())
      .then((s: SystemStatus) => {
        setStatus(s);
        if (s.default_country && s.countries?.some((c) => c.code === s.default_country)) {
          setCountry(s.default_country);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    const msgId = (Date.now() + 1).toString();
    const assistantMsg: ChatMessage = {
      id: msgId,
      role: 'assistant',
      content: '',
      reasoning: '',
      sources: [],
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.content, top_k: 5, country }),
      });
      if (!res.ok) throw new Error('Failed to query');

      const reader = res.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let reasoningDone = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.done) break;

            if (data.type === 'reasoning') {
              setMessages(prev =>
                prev.map(m =>
                  m.id === msgId
                    ? { ...m, reasoning: (m.reasoning || '') + data.content }
                    : m
                )
              );
            } else if (data.type === 'content') {
              if (!reasoningDone) reasoningDone = true;
              setMessages(prev =>
                prev.map(m =>
                  m.id === msgId
                    ? { ...m, content: (m.content || '') + data.content }
                    : m
                )
              );
            }
          } catch { /* skip malformed lines */ }
        }
      }
    } catch (error) {
      setMessages(prev =>
        prev.map(m =>
          m.id === msgId
            ? { ...m, content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}` }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🌱</span>
            <h1 className="text-lg font-bold text-gray-800">AusOrganicBot</h1>
            <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">Open Source</span>
          </div>
          {status && (
            <div className="flex items-center gap-4 text-sm">
              {status.countries && status.countries.length > 1 && (
                <select
                  value={country}
                  onChange={(e) => setCountry(e.target.value)}
                  className="px-2 py-1 border border-gray-300 rounded-md text-gray-700 bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  title="Which country's standards to answer from"
                >
                  {status.countries.map((c) => (
                    <option key={c.code} value={c.code}>
                      {c.code.toUpperCase()} — {c.name}
                    </option>
                  ))}
                </select>
              )}
              <span className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${status.llm_connected ? 'bg-green-500' : 'bg-red-500'}`} />
                {status.llm_connected ? 'Connected' : 'Disconnected'}
              </span>
              <span className="text-gray-500">{status.documents_loaded} standards loaded</span>
            </div>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-16">
                <div className="text-5xl mb-4">🌿</div>
                <h2 className="text-xl font-semibold text-gray-700 mb-2">
                  Australian Organic Certification Q&A
                </h2>
                <p className="text-gray-500 mb-6 max-w-md mx-auto">
                  Ask questions about organic farming standards, input approvals, buffer zones, record-keeping, and certification requirements.
                </p>
                <div className="flex flex-wrap gap-2 justify-center max-w-lg mx-auto">
                  {[
                    'What are buffer zone requirements between organic and conventional crops?',
                    'Can I use copper sulfate as a fungicide?',
                    'What records do I need for organic certification?',
                    'How do I transition from conventional to organic farming?',
                  ].map((q) => (
                    <button
                      key={q}
                      onClick={() => setInput(q)}
                      className="px-3 py-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-green-50 hover:border-green-300 hover:text-green-700 transition"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-xl px-4 py-3 ${
                  msg.role === 'user'
                    ? 'bg-green-600 text-white'
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}>
                  {/* Thinking / reasoning block */}
                  {msg.reasoning && (
                    <div className="mb-2 pb-2 border-b border-dashed border-gray-200">
                      <details open={isLoading && msg.content === ''}>
                        <summary className="text-xs text-amber-600 cursor-pointer hover:text-amber-700 select-none font-medium">
                          {msg.content === '' ? '⏳ Thinking...' : '🧠 Show thinking'}
                        </summary>
                        <div className="mt-1 p-2 bg-amber-50 rounded text-xs text-amber-800 whitespace-pre-wrap max-h-48 overflow-auto leading-relaxed">
                          {msg.reasoning}
                        </div>
                      </details>
                    </div>
                  )}
                  {/* Answer content */}
                  <div className="text-sm prose prose-sm max-w-none prose-a:text-green-600 prose-strong:text-gray-900 prose-code:text-amber-700">
                    {msg.content ? (
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    ) : msg.reasoning ? (
                      <span className="text-gray-400 italic">Generating answer...</span>
                    ) : null}
                  </div>
                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-gray-100">
                      <p className="text-xs font-medium text-gray-400 mb-1">Sources:</p>
                      {msg.sources.map((s, i) => (
                        <div key={i} className="bg-gray-50 p-2 rounded text-xs mb-1 last:mb-0">
                          <span className="font-medium text-green-600">
                            {formatSourceName(s.metadata.source || 'unknown')}
                          </span>
                          {s.metadata.edition && (
                            <span className="text-gray-400 ml-2">Ed. {s.metadata.edition}</span>
                          )}
                          <span className="text-gray-400 ml-2">
                            {s.metadata.filename ? `— ${s.metadata.filename}` : ''}
                          </span>
                          <span className="text-gray-400 ml-2">
                            {s.metadata.page_number ? `p.${s.metadata.page_number}` : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && messages.length === 0 && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
                  <span className="text-sm text-gray-500">Searching standards...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about organic standards..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent text-gray-900 text-base placeholder:text-gray-400 bg-white"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
