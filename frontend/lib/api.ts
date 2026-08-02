const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export async function checkStatus() {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function queryBackend(question: string, top_k = 5) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k }),
  });
  if (!res.ok) throw new Error('Failed to query');
  return res.json();
}

export async function* streamQuery(question: string, top_k = 5): AsyncGenerator<string> {
  const res = await fetch(`${API_BASE}/api/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k }),
  });
  if (!res.ok) throw new Error('Failed to stream query');
  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
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
        if (data.done) return;
        if (data.chunk) yield data.chunk;
      } catch { /* skip malformed lines */ }
    }
  }
}

export async function checkInput(product_name: string, ingredients: string[], intended_use: string) {
  const res = await fetch(`${API_BASE}/api/input-check`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product_name, ingredients, intended_use }),
  });
  if (!res.ok) throw new Error('Failed to check input');
  return res.json();
}

export async function getSources() {
  const res = await fetch(`${API_BASE}/api/sources`);
  if (!res.ok) throw new Error('Failed to fetch sources');
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.ok;
}
