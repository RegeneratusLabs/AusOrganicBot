"""Verify the streaming parser tolerates providers without reasoning_content (e.g. Ollama)."""
import json

sim_lines = [
    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
    'data: {"choices":[{"delta":{"content":" world"}}]}',
    'data: [DONE]',
]

results = []
for line in sim_lines:
    if not line.strip():
        continue
    if line.startswith("data: "):
        raw = line[6:]
        if raw == "[DONE]":
            break
        chunk = json.loads(raw)
        delta = chunk["choices"][0]["delta"]
        rc = delta.get("reasoning_content")  # None for Ollama-style providers
        if rc:
            results.append(("reasoning", rc))
        c = delta.get("content")
        if c:
            results.append(("content", c))

print("Parsed events:", results)
assert results == [("content", "Hello"), ("content", " world")], "parse mismatch"
print("OK: no reasoning_content field tolerated without error")
