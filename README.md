# ChromaDB vs. FAISS

The notebook retains example retrieval cells and runs the repeatable
exact-versus-approximate benchmark in its final cell. Run it directly with:

```powershell
uv run python benchmark.py
```

The benchmark loads the four Naruto episode JSON files, creates normalized
embeddings using the selected Sentence Transformers model, and precomputes
query embeddings before timing. It compares exact FAISS `IndexFlatIP` against
FAISS IVF, FAISS HNSW, and Chroma HNSW. Search configurations are run in
randomized paired order and report p50/p95 latency across repeated, warmed
queries. Embedding time is reported separately.

The starter qrels contain 50 hand-authored queries with known relevant
episode(s) graded 3. This is a useful smoke-test set, not a fully pooled
relevance study: unjudged episodes currently count as non-relevant. Use
`--pool-output` to export the systems' retrieved candidates, have group
members independently judge relevance from 0 to 3, resolve disagreements,
then use the completed CSV with `--qrels`. Metrics include labeled
Recall@k/MRR@k/nDCG@k and ANN recall@k against exact FAISS top-k; these answer
different questions.

The run also measures build/ingestion throughput, serialized/persistent disk
size, process working-set change, persistence reopen and first-query time,
metadata filtering, and Chroma update/delete versus FAISS full-index rebuild
cost. It runs scaling checks at 1x, 2x, and 5x by default. Above 1x the
dataset is repeated vectors with synthetic IDs; these runs are for load and
latency scaling only, not semantic quality. FAISS metadata filtering is
measured as an exact full-index scan followed by post-filtering, not as a
native database filter. Memory deltas are process-level measurements and may
include runtime overhead.

JSON, search CSV, and operational CSV artifacts are written to
`benchmark_results/` with model and software-version metadata.

Options:

```powershell
uv run python benchmark.py --repeats 30 --k 5 --target-recall 0.95 --model all-mpnet-base-v2
```

Export an annotation pool after the benchmark run:

```powershell
uv run python benchmark.py --pool-output evaluation_pool.csv
```

After annotating every pooled row with an integer relevance grade from 0
(irrelevant) to 3 (highly relevant), use the completed file as the qrels:

```powershell
uv run python benchmark.py --qrels evaluation_pool.csv --scale-multipliers 1,2,5
```

For a quick model comparison, run the command once per model, for example
`all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and
`multi-qa-MiniLM-L6-cos-v1`. Keep the qrels, `k`, and index settings the same
for every run. Model downloads may require internet access the first time.

The ephemeral ANN collections are removed after the search comparison. A
separate temporary persistent Chroma database is used for persistence and
operational tests and deleted when the run finishes.

Run the metric and configuration-selection tests with:

```powershell
python -m unittest discover -s tests -v
```
