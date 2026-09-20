# OrgStandardsBot — development and evaluation tasks
#
# Accuracy work lives behind `make bench`. The benchmark is human-adjudicated:
# `bench-build` creates drafts that are NOT scoreable, `bench-review` lets a
# human promote them to ground truth, and `bench-score` reports over verified
# items only. See ACCURACY.md.

PYTHON ?= python3
COUNTRY ?= au

# The embedding model is cached under .hf_cache (gitignored). Without this the
# backend and all eval tooling try to write to ~/.cache/huggingface.
export HF_HOME := $(CURDIR)/.hf_cache

.PHONY: help test test-backend bench bench-build bench-review bench-score \
        ingest serve clean-bench lint

help:
	@echo "OrgStandardsBot targets"
	@echo "  make test          run the backend test suite"
	@echo "  make ingest        ingest standards for COUNTRY (default au)"
	@echo "  make serve         run the backend API on :8080"
	@echo ""
	@echo "  Accuracy benchmark (human-adjudicated, see ACCURACY.md):"
	@echo "  make bench-build   write DRAFT items (not scoreable)"
	@echo "  make bench-review  interactively verify drafts as a named human"
	@echo "  make bench-score   score verified items and report with CIs"
	@echo "  make clean-bench   delete generated benchmark reports"

test:
	$(PYTHON) -m pytest backend/tests -q

test-backend: test

# Build drafts for COUNTRY. Overwrites only with FORCE=1, because verifying is
# lost human work.
bench-build:
	$(PYTHON) evals/build_benchmark.py --country $(COUNTRY) $(if $(FORCE),--force,)

bench-review:
	$(PYTHON) evals/review_benchmark.py --country $(COUNTRY) $(if $(REVIEWER),--reviewer "$(REVIEWER)",)

bench-score:
	$(PYTHON) evals/score_benchmark.py --country $(COUNTRY) $(if $(JUDGE),--judge,)

# Full loop: build drafts if absent, review, then score.
bench: bench-build bench-review bench-score

ingest:
	$(PYTHON) scripts/ingest.py --country $(COUNTRY) --no-html

serve:
	cd backend && $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8080

clean-bench:
	rm -f evals/data/benchmark_*.json
	@echo "Removed generated benchmark reports (item files in evals/bench/ kept)"
