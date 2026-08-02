export function formatSourceName(source: string): string {
  const names: Record<string, string> = {
    nasaa: 'NASAA',
    aco: 'ACO',
    sxo: 'SXO',
    national: 'National Standard',
  };
  return names[source] || source.toUpperCase();
}

export function formatConfidence(score: number): string {
  return `${Math.round(score * 100)}%`;
}

export function formatTime(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}
