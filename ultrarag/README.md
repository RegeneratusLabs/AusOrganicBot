# Running AusOrganicBot on UltraRAG (OpenBMB)

[UltraRAG](https://github.com/OpenBMB/UltraRAG) (Apache-2.0, THUNLP +
OpenBMB + NEUIR) is the OpenBMB RAG framework — MCP servers composed into
YAML pipelines, with a built-in evaluation system. `ausorganicbot.yaml`
re-implements the app's core logic (retrieve → relevance gate → generate /
refuse) on UltraRAG with the OpenBMB model stack.

## Stack

| Component | App (default) | UltraRAG pipeline |
|---|---|---|
| Retriever | all-MiniLM-L6-v2 + ChromaDB | **MiniCPM-Embedding-Light** (dense+sparse) + Milvus |
| Reader | any OpenAI-compatible LLM | **MiniCPM5-1B** (vLLM/SGLang/Ollama) |
| Gate | hardcoded 0.50 similarity | `prompt.check_passages` branch → refusal tool |

## Setup

```bash
# 1. Install UltraRAG (venv recommended)
pip install uv && git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG && uv sync --all-extras && source .venv/bin/activate

# 2. Serve MiniCPM5-1B (OpenAI-compatible, port 8000)
vllm serve openbmb/MiniCPM5-1B --port 8000

# 3. Serve MiniCPM-Embedding-Light as an embedding endpoint
#    (vLLM --task embed, port 8001, or the bundled Infinity server)

# 4. Start Milvus (standalone docker)
#    see https://ultrarag.openbmb.cn/pages/en/ui/prepare

# 5. Copy this directory into UltraRAG's examples tree and index the corpus
cp ../AusOrganicBot/ultrarag/ausorganicbot.yaml examples/experiments/
#    Index data/<cc>/html/full.html (or the PDFs) into Milvus with the
#    retriever_init/retriever_index tools (see examples/demos/milvus_index.yaml)

# 6. Run the pipeline
ultrarag run examples/experiments/ausorganicbot.yaml
```

## Parameter notes

UltraRAG configures models per-experiment in a `parameter/` directory next
to the YAML (endpoint URLs, model names, retrieval top-k and score
threshold). Point the retriever at the MiniCPM-Embedding-Light endpoint
and the generation server at the MiniCPM5-1B endpoint, set the score
threshold to 0.50 (or the per-country threshold from
`data/<cc>/manifest.json`), and `prompt.check_passages` implements the
same gate as the app.

The YAML was written against UltraRAG 3.0 examples
(`examples/experiments/{vanilla_rag,rag_branch}.yaml`); tool names may
drift between releases — adjust against the examples shipped with your
checkout.

## Evaluation

UltraRAG's evaluation server can consume the same gold set the app uses
(`evals/data/gold_<cc>.jsonl`) for retrieval metrics, and
`evals/llm_judge.py` can grade UltraRAG-generated answers by pointing the
app at the same MiniCPM endpoints.

## Why keep both?

- **ChromaDB app** — zero-infra, offline-friendly, batteries included;
  the reference implementation and default experience
- **UltraRAG pipeline** — the research/production-grade path: Milvus at
  scale, MCP composability, built-in benchmarks, and the full OpenBMB
  ecosystem (MiniCPM-Embedding-Light + MiniCPM-Reranker-Light pair
  recommended in its model card)
