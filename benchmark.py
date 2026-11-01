import argparse
import csv
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


DATA_FILES = (
    "naruto_s1_backup.json",
    "naruto_s2.json",
    "naruto_s2_secondpart.json",
    "naruto_s2_thirdpart.json",
)


def mean_recall_at_k(
    results: dict[str, Sequence[str]],
    exact_results: dict[str, Sequence[str]],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    recalls = []
    for query, exact_ids in exact_results.items():
        expected = set(exact_ids[:k])
        if expected:
            recalls.append(len(expected.intersection(results[query][:k])) / len(expected))
    if not recalls:
        raise ValueError("Cannot calculate recall without exact results")
    return statistics.fmean(recalls)


def labeled_metrics(
    results: dict[str, Sequence[str]],
    relevance: dict[str, set[str]],
    k: int,
) -> tuple[float, float]:
    if k < 1:
        raise ValueError("k must be at least 1")
    recalls = []
    reciprocal_ranks = []
    for query, relevant_ids in relevance.items():
        if not relevant_ids:
            raise ValueError(f"Query has no relevance judgments: {query}")
        retrieved = list(results[query][:k])
        recalls.append(len(set(retrieved).intersection(relevant_ids)) / len(relevant_ids))
        reciprocal_ranks.append(
            next(
                (1 / rank for rank, result_id in enumerate(retrieved, start=1) if result_id in relevant_ids),
                0.0,
            )
        )
    return statistics.fmean(recalls), statistics.fmean(reciprocal_ranks)


def ndcg_at_k(
    results: dict[str, Sequence[str]],
    judgments: dict[str, dict[str, int]],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    scores = []
    for query, grades in judgments.items():
        if not any(grade > 0 for grade in grades.values()):
            raise ValueError(f"Query has no relevant judgments: {query}")

        def dcg(episode_ids: Sequence[str]) -> float:
            return sum(
                (2**grades.get(episode_id, 0) - 1) / math.log2(rank + 1)
                for rank, episode_id in enumerate(episode_ids[:k], start=1)
            )

        ideal_ids = sorted(grades, key=grades.get, reverse=True)
        ideal_dcg = dcg(ideal_ids)
        scores.append(dcg(results[query]) / ideal_dcg if ideal_dcg else 0.0)
    if not scores:
        raise ValueError("Cannot calculate nDCG without judgments")
    return statistics.fmean(scores)


def load_relevance_judgments(
    path: Path,
    known_ids: set[str],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    judgments: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"query", "episode_id", "relevance"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must have columns: query, episode_id, relevance"
            )
        for line_number, row in enumerate(reader, start=2):
            query = (row["query"] or "").strip()
            episode_id = (row["episode_id"] or "").strip()
            try:
                grade = int(row["relevance"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: relevance must be a non-negative integer"
                ) from error
            if not query or not episode_id or not 0 <= grade <= 3:
                raise ValueError(
                    f"{path}:{line_number}: query and episode_id are required; "
                    "relevance must be an integer from 0 to 3"
                )
            if episode_id not in known_ids:
                raise ValueError(
                    f"{path}:{line_number}: unknown episode_id {episode_id!r}"
                )
            query_judgments = judgments.setdefault(query, {})
            if episode_id in query_judgments:
                raise ValueError(
                    f"{path}:{line_number}: duplicate query/episode judgment"
                )
            query_judgments[episode_id] = grade

    for query, grades in judgments.items():
        if not any(grade > 0 for grade in grades.values()):
            raise ValueError(f"Query has no relevant episode judgment: {query}")
    if not judgments:
        raise ValueError(f"No relevance judgments found in {path}")
    return list(judgments), judgments


def select_matched_configurations(
    sweeps: dict[str, list[dict]],
    requested_recall: float,
) -> tuple[float, dict[str, dict]]:
    if not 0 < requested_recall <= 1:
        raise ValueError("requested_recall must be in (0, 1]")
    if not sweeps or any(not configurations for configurations in sweeps.values()):
        raise ValueError("Every backend must have at least one measured configuration")

    matched_recall = min(
        requested_recall,
        *(max(configuration["recall"] for configuration in configurations)
          for configurations in sweeps.values()),
    )
    selected = {}
    for backend, configurations in sweeps.items():
        eligible = [
            configuration
            for configuration in configurations
            if configuration["recall"] + 1e-12 >= matched_recall
        ]
        selected[backend] = min(eligible, key=lambda configuration: configuration["p50_ms"])
    return matched_recall, selected


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("At least one timing sample is required")
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure_paired_searches(
    searchers: dict[str, Callable[[object], list[str]]],
    query_vectors: Sequence[object],
    queries: Sequence[str],
    repeats: int,
    k: int,
    seed: int,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[float]]]:
    rng = random.Random(seed)
    results = {name: {} for name in searchers}
    timings = {name: [] for name in searchers}
    for search in searchers.values():
        for vector in query_vectors:
            search(vector)
    for _ in range(repeats):
        for query, vector in zip(queries, query_vectors):
            order = list(searchers)
            rng.shuffle(order)
            for name in order:
                start = time.perf_counter()
                found = searchers[name](vector)
                timings[name].append((time.perf_counter() - start) * 1000)
                results[name][query] = found[:k]
    return results, timings


def _process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if success else None
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_run_artifacts(
    output_dir: Path,
    summary: dict,
    configurations: list[dict],
    operations: dict,
    scaling: list[dict],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    json_path = output_dir / f"benchmark_{stamp}.json"
    csv_path = output_dir / f"benchmark_{stamp}.csv"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "configurations": configurations,
                "operations": operations,
                "scaling": scaling,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    versions = summary["versions"]
    fields = [
        "created_at_utc",
        "model",
        "python_version",
        "platform",
        "numpy_version",
        "faiss_version",
        "chromadb_version",
        "sentence_transformers_version",
        "k",
        "repeats",
        "dataset_size",
        "dataset_type",
        "record_type",
        "backend",
        "setting",
        "recall_at_k",
        "p50_ms",
        "p95_ms",
        "labeled_recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "build_seconds",
        "ingest_docs_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for configuration in configurations:
            writer.writerow(
                {
                    "created_at_utc": summary["created_at_utc"],
                    "model": summary["model"],
                    "python_version": versions["python"],
                    "platform": versions["platform"],
                    "numpy_version": versions["numpy"],
                    "faiss_version": versions["faiss-cpu"],
                    "chromadb_version": versions["chromadb"],
                    "sentence_transformers_version": versions["sentence-transformers"],
                    "k": summary["k"],
                    "repeats": summary["repeats"],
                    **configuration,
                }
            )
    operations_path = output_dir / f"benchmark_operations_{stamp}.csv"
    operation_rows = []
    for backend, metrics in operations.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    operation_rows.extend(
                        {
                            "backend": backend,
                            "operation": f"{key}.{nested_key}",
                            "value": nested_value,
                        }
                        for nested_key, nested_value in value.items()
                        if not isinstance(nested_value, (dict, list))
                    )
                elif not isinstance(value, list):
                    operation_rows.append(
                        {"backend": backend, "operation": key, "value": value}
                    )
    with operations_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "created_at_utc",
                "model",
                "python_version",
                "chromadb_version",
                "faiss_version",
                "backend",
                "operation",
                "value",
            ],
        )
        writer.writeheader()
        run_metadata = {
            "created_at_utc": summary["created_at_utc"],
            "model": summary["model"],
            "python_version": versions["python"],
            "chromadb_version": versions["chromadb"],
            "faiss_version": versions["faiss-cpu"],
        }
        writer.writerows({**run_metadata, **row} for row in operation_rows)
        for result in scaling:
            for backend in ("faiss_flat", "chroma_hnsw"):
                for metric, value in result[backend].items():
                    writer.writerow(
                        {
                            "backend": backend,
                            "operation": f"scale_{result['dataset_size']}_{metric}",
                            "value": value,
                        }
                    )
    return json_path, csv_path, operations_path


def _export_judgment_pool(
    output_path: Path,
    queries: Sequence[str],
    exact_results: dict[str, Sequence[str]],
    configurations: Sequence[dict],
    judgments: dict[str, dict[str, int]],
) -> None:
    pool: dict[str, dict[str, set[str]]] = {
        query: {} for query in queries
    }
    for query in queries:
        for episode_id in judgments.get(query, {}):
            pool[query].setdefault(episode_id, set()).add("starter qrels")
        for episode_id in exact_results[query]:
            pool[query].setdefault(episode_id, set()).add("FAISS exact")
    for configuration in configurations:
        for query, episode_ids in configuration["results"].items():
            for episode_id in episode_ids:
                pool[query].setdefault(episode_id, set()).add(
                    f"{configuration['backend']} {configuration['setting']}"
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query", "episode_id", "relevance", "retrieved_by"],
        )
        writer.writeheader()
        for query in queries:
            for episode_id, sources in sorted(pool[query].items()):
                known_grade = judgments.get(query, {}).get(episode_id, "")
                writer.writerow(
                    {
                        "query": query,
                        "episode_id": episode_id,
                        "relevance": known_grade,
                        "retrieved_by": "; ".join(sorted(sources)),
                    }
                )


def _validate_scale_multipliers(multipliers: Sequence[int]) -> None:
    if not multipliers or any(multiplier < 1 for multiplier in multipliers):
        raise ValueError("At least one positive scale multiplier is required")


def _measure_operational_behavior(
    ids: Sequence[str],
    documents: Sequence[str],
    embeddings,
    records: Sequence[dict],
    query_vector,
    k: int,
) -> dict:
    import chromadb
    import faiss
    import numpy as np

    dimension = embeddings.shape[1]
    metadata = [
        {
            "url": record["url"],
            "season": 1 if int(record["id"]) <= 220 else 2,
        }
        for record in records
    ]
    report = {}
    with tempfile.TemporaryDirectory(prefix="vector_benchmark_") as temporary:
        root = Path(temporary)
        rss_before = _process_rss_bytes()
        flat = faiss.IndexFlatIP(dimension)
        build_start = time.perf_counter()
        flat.add(embeddings)
        faiss_path = root / "faiss.index"
        faiss.write_index(flat, str(faiss_path))
        faiss_metadata_path = root / "faiss_metadata.json"
        faiss_metadata_path.write_text(
            json.dumps(
                {
                    "ids": list(ids),
                    "documents": list(documents),
                    "metadata": metadata,
                }
            ),
            encoding="utf-8",
        )
        faiss_build_seconds = time.perf_counter() - build_start
        rss_after = _process_rss_bytes()
        restart_start = time.perf_counter()
        reloaded_flat = faiss.read_index(str(faiss_path))
        json.loads(faiss_metadata_path.read_text(encoding="utf-8"))
        faiss_restart_seconds = time.perf_counter() - restart_start
        faiss_restart_query_start = time.perf_counter()
        reloaded_flat.search(query_vector.reshape(1, -1), k)
        faiss_restart_first_query_ms = (
            time.perf_counter() - faiss_restart_query_start
        ) * 1000
        report["faiss"] = {
            "build_seconds": faiss_build_seconds,
            "ingest_docs_per_second": len(ids) / max(faiss_build_seconds, 1e-12),
            "serialized_bytes": faiss_path.stat().st_size,
            "metadata_sidecar_bytes": faiss_metadata_path.stat().st_size,
            "total_persisted_bytes": _directory_size(root),
            "restart_load_seconds": faiss_restart_seconds,
            "restart_first_query_ms": faiss_restart_first_query_ms,
            "process_rss_delta_bytes": (
                max(0, rss_after - rss_before)
                if rss_before is not None and rss_after is not None
                else None
            ),
        }
        updated_vectors = embeddings.copy()
        updated_vectors[0] = updated_vectors[0] + np.float32(0.01)
        updated_vectors[0] /= np.linalg.norm(updated_vectors[0])
        update_rebuild_start = time.perf_counter()
        updated_index = faiss.IndexFlatIP(dimension)
        updated_index.add(updated_vectors)
        faiss_update_rebuild_seconds = time.perf_counter() - update_rebuild_start
        delete_rebuild_start = time.perf_counter()
        deleted_index = faiss.IndexFlatIP(dimension)
        deleted_index.add(embeddings[1:])
        faiss_delete_rebuild_seconds = time.perf_counter() - delete_rebuild_start
        report["faiss"]["update_rebuild_seconds"] = faiss_update_rebuild_seconds
        report["faiss"]["delete_rebuild_seconds"] = faiss_delete_rebuild_seconds
        report["faiss"]["crud_method"] = "full IndexFlat rebuild after one update/delete"

        persistent_path = root / "chroma"
        chroma_client = chromadb.PersistentClient(path=str(persistent_path))
        chroma_rss_before = _process_rss_bytes()
        ingest_start = time.perf_counter()
        collection = chroma_client.create_collection(
            name="operations",
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 64,
                    "max_neighbors": 16,
                    "ef_construction": 100,
                }
            },
        )
        collection.add(
            ids=list(ids),
            embeddings=embeddings.tolist(),
            documents=list(documents),
            metadatas=metadata,
        )
        chroma_ingest_seconds = time.perf_counter() - ingest_start
        chroma_rss_after = _process_rss_bytes()
        chroma_bytes = _directory_size(persistent_path)
        del collection
        chroma_client.close()
        del chroma_client
        gc.collect()
        restart_start = time.perf_counter()
        chroma_client = chromadb.PersistentClient(path=str(persistent_path))
        collection = chroma_client.get_collection("operations")
        restarted_count = collection.count()
        chroma_restart_seconds = time.perf_counter() - restart_start
        first_query_start = time.perf_counter()
        collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            include=["distances"],
        )
        chroma_restart_first_query_ms = (
            time.perf_counter() - first_query_start
        ) * 1000

        filter_start = time.perf_counter()
        filtered = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where={"season": 1},
            include=["metadatas"],
        )
        chroma_filter_ms = (time.perf_counter() - filter_start) * 1000
        if any(item["season"] != 1 for item in filtered["metadatas"][0]):
            raise RuntimeError("Chroma metadata filter returned an out-of-filter result")

        scratch_id = "__benchmark_crud_record__"
        collection.add(
            ids=[scratch_id],
            embeddings=[embeddings[0].tolist()],
            documents=["temporary benchmark record"],
            metadatas=[metadata[0]],
        )
        update_start = time.perf_counter()
        collection.update(
            ids=[scratch_id],
            documents=["updated benchmark record"],
        )
        chroma_update_ms = (time.perf_counter() - update_start) * 1000
        delete_start = time.perf_counter()
        collection.delete(ids=[scratch_id])
        chroma_delete_ms = (time.perf_counter() - delete_start) * 1000

        faiss_filter_start = time.perf_counter()
        _, all_positions = reloaded_flat.search(
            query_vector.reshape(1, -1), len(ids)
        )
        faiss_filtered_positions = [
            int(position)
            for position in all_positions[0]
            if position >= 0 and metadata[int(position)]["season"] == 1
        ][:k]
        faiss_filter_ms = (time.perf_counter() - faiss_filter_start) * 1000

        report["chroma"] = {
            "build_and_ingest_seconds": chroma_ingest_seconds,
            "ingest_docs_per_second": len(ids) / max(chroma_ingest_seconds, 1e-12),
            "persistent_bytes": chroma_bytes,
            "restart_reopen_and_count_seconds": chroma_restart_seconds,
            "restart_first_query_ms": chroma_restart_first_query_ms,
            "count_after_restart": restarted_count,
            "process_rss_delta_bytes": (
                max(0, chroma_rss_after - chroma_rss_before)
                if chroma_rss_before is not None and chroma_rss_after is not None
                else None
            ),
            "metadata_filter_ms": chroma_filter_ms,
            "update_ms": chroma_update_ms,
            "delete_ms": chroma_delete_ms,
        }
        report["metadata_filter_comparison"] = {
            "filter": "season=1",
            "k": k,
            "chroma_latency_ms": chroma_filter_ms,
            "faiss_full_scan_and_postfilter_latency_ms": faiss_filter_ms,
            "chroma_result_count": len(filtered["ids"][0]),
            "faiss_result_count": len(faiss_filtered_positions),
            "faiss_method": "exact full-index scan followed by metadata filtering",
        }
        del collection
        chroma_client.close()
    return report


def _run_scaling_experiments(
    ids: Sequence[str],
    embeddings,
    query_vectors,
    multipliers: Sequence[int],
    queries: Sequence[str],
    k: int,
    repeats: int,
    seed: int,
) -> list[dict]:
    import chromadb
    import faiss
    import numpy as np

    reports = []
    for multiplier in multipliers:
        if multiplier < 1:
            raise ValueError("Scale multipliers must be positive integers")
        scaled_embeddings = np.tile(embeddings, (multiplier, 1))
        scaled_ids = [
            episode_id if copy == 0 else f"{episode_id}__synthetic_{copy}"
            for copy in range(multiplier)
            for episode_id in ids
        ]
        flat = faiss.IndexFlatIP(scaled_embeddings.shape[1])
        faiss_start = time.perf_counter()
        flat.add(scaled_embeddings)
        faiss_build_seconds = time.perf_counter() - faiss_start

        chroma_client = chromadb.EphemeralClient()
        chroma_start = time.perf_counter()
        collection = chroma_client.create_collection(
            name=f"scale_{multiplier}",
            configuration={"hnsw": {"space": "cosine"}},
        )
        collection.add(
            ids=scaled_ids,
            embeddings=scaled_embeddings.tolist(),
        )
        chroma_build_seconds = time.perf_counter() - chroma_start

        def exact_search(vector):
            _, positions = flat.search(vector.reshape(1, -1), k)
            return [scaled_ids[int(position)] for position in positions[0] if position >= 0]

        def chroma_search(vector):
            result = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=k,
                include=["distances"],
            )
            return result["ids"][0]

        _, timings = _measure_paired_searches(
            {"FAISS Flat": exact_search, "Chroma HNSW": chroma_search},
            query_vectors,
            queries,
            repeats,
            k,
            seed + multiplier,
        )
        reports.append(
            {
                "dataset_size": len(scaled_ids),
                "dataset_type": "real" if multiplier == 1 else "synthetic_repeated_vectors",
                "multiplier": multiplier,
                "faiss_flat": {
                    "build_seconds": faiss_build_seconds,
                    "p50_ms": _percentile(timings["FAISS Flat"], 50),
                    "p95_ms": _percentile(timings["FAISS Flat"], 95),
                },
                "chroma_hnsw": {
                    "build_seconds": chroma_build_seconds,
                    "p50_ms": _percentile(timings["Chroma HNSW"], 50),
                    "p95_ms": _percentile(timings["Chroma HNSW"], 95),
                },
            }
        )
        chroma_client.delete_collection(f"scale_{multiplier}")
        chroma_client.close()
    return reports


def run_benchmark(
    repeats: int = 20,
    k: int = 5,
    target_recall: float = 0.95,
    model_name: str = "all-MiniLM-L6-v2",
    qrels_path: Path | None = None,
    output_dir: Path | None = None,
    pool_output: Path | None = None,
    seed: int = 42,
    scale_multipliers: Sequence[int] = (1, 2, 5),
) -> None:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    _validate_scale_multipliers(scale_multipliers)

    import chromadb
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    root = Path(__file__).resolve().parent
    records = []
    for filename in DATA_FILES:
        with (root / filename).open(encoding="utf-8") as file:
            records.extend(json.load(file))

    ids = [str(record["id"]) for record in records]
    documents = [record["raw_text"] for record in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("The benchmark dataset must be non-empty and have unique IDs")
    if k > len(records):
        raise ValueError(f"k={k} exceeds the {len(records)} available records")
    qrels_path = qrels_path or root / "evaluation_qrels.csv"
    if pool_output is not None and pool_output.resolve() == qrels_path.resolve():
        raise ValueError("--pool-output must not overwrite the input --qrels file")
    queries, judgments = load_relevance_judgments(qrels_path, set(ids))
    relevance = {
        query: {episode_id for episode_id, grade in grades.items() if grade > 0}
        for query, grades in judgments.items()
    }

    model = SentenceTransformer(model_name)
    embedding_start = time.perf_counter()
    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    document_embedding_seconds = time.perf_counter() - embedding_start
    query_embedding_start = time.perf_counter()
    query_vectors = model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    query_embedding_seconds = time.perf_counter() - query_embedding_start

    dimension = embeddings.shape[1]
    exact_index = faiss.IndexFlatIP(dimension)
    exact_build_start = time.perf_counter()
    exact_index.add(embeddings)
    exact_build_seconds = time.perf_counter() - exact_build_start

    def faiss_search(index, vector):
        _, positions = index.search(vector.reshape(1, -1), k)
        return [ids[position] for position in positions[0] if position >= 0]

    exact_search = lambda vector: faiss_search(exact_index, vector)
    exact_results = {
        query: exact_search(vector)
        for query, vector in zip(queries, query_vectors)
    }
    sweeps: dict[str, list[dict]] = {}

    def record_configuration(backend, setting, search):
        candidate_results = {
            query: search(vector)
            for query, vector in zip(queries, query_vectors)
        }
        recall = mean_recall_at_k(candidate_results, exact_results, k)
        configuration = {
            "backend": backend,
            "setting": setting,
            "recall": recall,
            "search": search,
            "candidate_results": candidate_results,
        }
        sweeps.setdefault(backend, []).append(configuration)

    count = len(records)
    nlist = max(1, min(int(math.sqrt(count)), count // 39 or 1))
    quantizer = faiss.IndexFlatIP(dimension)
    ivf_index = faiss.IndexIVFFlat(
        quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
    )
    ivf_build_start = time.perf_counter()
    ivf_index.train(embeddings)
    ivf_index.add(embeddings)
    ivf_build_seconds = time.perf_counter() - ivf_build_start
    nprobe_values = sorted(
        {1, max(1, nlist // 8), max(1, nlist // 4), max(1, nlist // 2), nlist}
    )
    for nprobe in nprobe_values:
        def ivf_search(vector, index=ivf_index, current_nprobe=nprobe):
            index.nprobe = current_nprobe
            return faiss_search(index, vector)

        record_configuration(
            "FAISS IVF",
            f"nprobe={nprobe}",
            ivf_search,
        )

    hnsw_index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = 100
    hnsw_build_start = time.perf_counter()
    hnsw_index.add(embeddings)
    hnsw_build_seconds = time.perf_counter() - hnsw_build_start
    ef_search_values = sorted({max(8, k), 16, 32, 64, 128, max(256, count)})
    for ef_search in ef_search_values:
        def hnsw_search(vector, index=hnsw_index, current_ef=ef_search):
            index.hnsw.efSearch = current_ef
            return faiss_search(index, vector)

        record_configuration(
            "FAISS HNSW",
            f"ef_search={ef_search}",
            hnsw_search,
        )

    chroma_client = chromadb.EphemeralClient()
    chroma_collection_names = []
    for ef_search in ef_search_values:
        collection_name = f"benchmark_chroma_{ef_search}"
        chroma_collection_names.append(collection_name)
        chroma_build_start = time.perf_counter()
        collection = chroma_client.create_collection(
            name=collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": ef_search,
                    "max_neighbors": 16,
                    "ef_construction": 100,
                }
            },
        )
        collection.add(ids=ids, embeddings=embeddings.tolist())
        chroma_build_seconds = time.perf_counter() - chroma_build_start

        def chroma_search(vector, current_collection=collection):
            result = current_collection.query(
                query_embeddings=[vector.tolist()],
                n_results=k,
                include=["distances"],
            )
            return result["ids"][0]

        record_configuration(
            "Chroma HNSW",
            f"ef_search={ef_search}",
            chroma_search,
        )
        sweeps["Chroma HNSW"][-1]["build_seconds"] = chroma_build_seconds

    build_seconds_by_backend = {
        "FAISS IVF": ivf_build_seconds,
        "FAISS HNSW": hnsw_build_seconds,
    }
    for backend, configurations in sweeps.items():
        for configuration in configurations:
            configuration.setdefault("build_seconds", build_seconds_by_backend.get(backend))

    all_searchers = {"FAISS Flat": exact_search}
    for backend, configurations in sweeps.items():
        for configuration in configurations:
            label = f"{backend} {configuration['setting']}"
            all_searchers[label] = configuration["search"]
    try:
        paired_results, paired_timings = _measure_paired_searches(
            all_searchers, query_vectors, queries, repeats, k, seed
        )
    finally:
        for collection_name in chroma_collection_names:
            chroma_client.delete_collection(collection_name)
        chroma_client.close()
    exact_timed_results = paired_results["FAISS Flat"]
    exact_timings = paired_timings["FAISS Flat"]
    configurations = []
    for backend, backend_configurations in sweeps.items():
        for configuration in backend_configurations:
            label = f"{backend} {configuration['setting']}"
            configuration["results"] = paired_results[label]
            configuration["p50_ms"] = _percentile(paired_timings[label], 50)
            configuration["p95_ms"] = _percentile(paired_timings[label], 95)
            configuration["ingest_docs_per_second"] = (
                count / configuration["build_seconds"]
                if configuration["build_seconds"]
                else None
            )
            configurations.append(configuration)
    exact_configuration = {
        "backend": "FAISS Flat",
        "setting": "exact",
        "recall": 1.0,
        "results": exact_timed_results,
        "p50_ms": _percentile(exact_timings, 50),
        "p95_ms": _percentile(exact_timings, 95),
        "build_seconds": exact_build_seconds,
        "ingest_docs_per_second": count / max(exact_build_seconds, 1e-12),
    }

    matched_recall, selected = select_matched_configurations(sweeps, target_recall)

    print(f"\nDataset: {count} documents, {dimension}-dimensional normalized vectors")
    print(f"Embedding model: {model_name}")
    print(f"Relevance judgments: {qrels_path} ({len(queries)} queries)")
    print(f"Query embeddings: precomputed once in {query_embedding_seconds:.3f}s")
    print(f"Document embeddings: {document_embedding_seconds:.3f}s (excluded from search timings)")
    print(f"Search repetitions: {repeats} per query and configuration; k={k}")
    print(f"Requested ANN recall@{k}: {target_recall:.3f}")
    print(f"Matched ANN recall@{k}:   {matched_recall:.3f}")
    print("Recall means overlap with FAISS exact top-k, averaged across queries.")
    print("Search timings are randomized paired warm runs; each query is pre-embedded.")
    print("\nConfiguration sweep (search-only latency):")
    print(
        f"{'Backend':<14} {'Setting':<20} {'Recall':>8} {'p50 ms':>10} "
        f"{'p95 ms':>10} {'Build s':>10} {'Docs/s':>10} {'Selected':>10}"
    )
    for backend, configurations in sweeps.items():
        for configuration in configurations:
            chosen = selected[backend] is configuration
            print(
                f"{backend:<14} {configuration['setting']:<20} "
                f"{configuration['recall']:>8.3f} {configuration['p50_ms']:>10.3f} "
                f"{configuration['p95_ms']:>10.3f} "
                f"{configuration['build_seconds']:>10.3f} "
                f"{configuration['ingest_docs_per_second']:>10.1f} "
                f"{('yes' if chosen else ''):>10}"
            )

    print(f"\nExact FAISS Flat (recall@{k}=1.000):")
    print(
        f"  p50={_percentile(exact_timings, 50):.3f} ms, "
        f"p95={_percentile(exact_timings, 95):.3f} ms, "
        f"build={exact_build_seconds:.3f}s "
        f"({count / max(exact_build_seconds, 1e-12):.1f} docs/s)"
    )
    exact_recall, exact_mrr = labeled_metrics(exact_timed_results, relevance, k)
    exact_ndcg = ndcg_at_k(exact_timed_results, judgments, k)
    print(
        f"  labeled Recall@{k}={exact_recall:.3f}, "
        f"MRR@{k}={exact_mrr:.3f}, nDCG@{k}={exact_ndcg:.3f}"
    )
    print(f"\nSelected ANN configurations at matched recall@{k}:")
    for backend, configuration in selected.items():
        recall, mrr = labeled_metrics(configuration["results"], relevance, k)
        ndcg = ndcg_at_k(configuration["results"], judgments, k)
        print(
            f"  {backend} ({configuration['setting']}): "
            f"ANN recall={configuration['recall']:.3f}, "
            f"p50={configuration['p50_ms']:.3f} ms, "
            f"p95={configuration['p95_ms']:.3f} ms, "
            f"labeled Recall@{k}={recall:.3f}, MRR@{k}={mrr:.3f}, "
            f"nDCG@{k}={ndcg:.3f}"
        )
    print(
        f"\nHuman-curated relevance judgments: "
        f"{sum(len(grades) for grades in judgments.values())} judged query/episode pairs. "
        "Unjudged episodes count as non-relevant; expand the qrels before "
        "drawing broad relevance conclusions."
    )

    operations = _measure_operational_behavior(
        ids, documents, embeddings, records, query_vectors[0], k
    )
    scaling = _run_scaling_experiments(
        ids, embeddings, query_vectors,
        scale_multipliers, queries, k, repeats, seed,
    )
    print("\nOperational measurements:")
    print(
        f"  Chroma ingest={operations['chroma']['build_and_ingest_seconds']:.3f}s "
        f"({operations['chroma']['ingest_docs_per_second']:.1f} docs/s), "
        f"disk={operations['chroma']['persistent_bytes']:,} bytes, "
        f"restart={operations['chroma']['restart_reopen_and_count_seconds']:.3f}s, "
        f"first query after restart={operations['chroma']['restart_first_query_ms']:.3f}ms"
    )
    print(
        f"  FAISS Flat build={operations['faiss']['build_seconds']:.3f}s "
        f"({operations['faiss']['ingest_docs_per_second']:.1f} docs/s), "
        f"serialized={operations['faiss']['serialized_bytes']:,} bytes, "
        f"vectors+metadata={operations['faiss']['total_persisted_bytes']:,} bytes, "
        f"restart={operations['faiss']['restart_load_seconds']:.3f}s, "
        f"first query after restart={operations['faiss']['restart_first_query_ms']:.3f}ms"
    )
    print(
        f"  Metadata filter season=1: Chroma={operations['metadata_filter_comparison']['chroma_latency_ms']:.3f}ms; "
        f"FAISS full scan + post-filter="
        f"{operations['metadata_filter_comparison']['faiss_full_scan_and_postfilter_latency_ms']:.3f}ms"
    )
    print(
        f"  Chroma update={operations['chroma']['update_ms']:.3f}ms, "
        f"delete={operations['chroma']['delete_ms']:.3f}ms; "
        f"FAISS update rebuild={operations['faiss']['update_rebuild_seconds']:.3f}s, "
        f"delete rebuild={operations['faiss']['delete_rebuild_seconds']:.3f}s"
    )
    print("\nScaling experiment (repeated vectors are synthetic; no relevance scores):")
    for result in scaling:
        print(
            f"  {result['dataset_size']} vectors ({result['dataset_type']}): "
            f"FAISS Flat p50={result['faiss_flat']['p50_ms']:.3f}ms; "
            f"Chroma HNSW p50={result['chroma_hnsw']['p50_ms']:.3f}ms"
        )

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "faiss-cpu": _package_version("faiss-cpu"),
        "chromadb": _package_version("chromadb"),
        "sentence-transformers": _package_version("sentence-transformers"),
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_size": count,
        "dataset_type": "real",
        "embedding_dimension": dimension,
        "model": model_name,
        "query_count": len(queries),
        "k": k,
        "repeats": repeats,
        "target_ann_recall": target_recall,
        "matched_ann_recall": matched_recall,
        "random_seed": seed,
        "qrels_path": str(qrels_path),
        "versions": versions,
        "timing_method": "randomized paired warm queries; p50/p95; embeddings excluded",
        "relevance_caveat": (
            "Starter qrels list known positive episodes; unjudged results are "
            "implicitly treated as non-relevant. Use the pooled annotation CSV "
            "to add grades 0-3 before drawing strong conclusions."
        ),
        "document_embedding_seconds": document_embedding_seconds,
        "query_embedding_seconds": query_embedding_seconds,
    }
    artifact_configurations = [exact_configuration] + configurations
    for configuration in artifact_configurations:
        labeled_recall, reciprocal_rank = labeled_metrics(
            configuration["results"], relevance, k
        )
        configuration["labeled_recall_at_k"] = labeled_recall
        configuration["mrr_at_k"] = reciprocal_rank
        configuration["ndcg_at_k"] = ndcg_at_k(
            configuration["results"], judgments, k
        )
        configuration["dataset_size"] = count
        configuration["dataset_type"] = "real"
        configuration["record_type"] = "search_configuration"
    artifact_configurations = [
        {key: value for key, value in configuration.items() if key != "search"}
        for configuration in artifact_configurations
    ]
    output_dir = output_dir or root / "benchmark_results"
    json_path, csv_path, operations_path = _write_run_artifacts(
        output_dir, summary, artifact_configurations, operations, scaling
    )
    print(
        f"\nSaved machine-readable results:\n  {json_path}\n  {csv_path}\n  {operations_path}"
    )
    if pool_output is not None:
        _export_judgment_pool(
            pool_output, queries, exact_results, configurations, judgments
        )
        print(f"Saved pooled annotation template: {pool_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact and approximate vector search.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scale-multipliers",
        default="1,2,5",
        help="Comma-separated real/synthetic corpus multipliers (e.g. 1,2,5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
    )
    parser.add_argument(
        "--pool-output",
        type=Path,
        help="Optional CSV path for pooled candidate relevance annotation",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path(__file__).resolve().parent / "evaluation_qrels.csv",
        help="CSV with query, episode_id, relevance columns",
    )
    args = parser.parse_args()
    try:
        scale_multipliers = tuple(
            int(value.strip())
            for value in args.scale_multipliers.split(",")
            if value.strip()
        )
    except ValueError as error:
        parser.error(f"--scale-multipliers must be comma-separated integers: {error}")
    run_benchmark(
        repeats=args.repeats,
        k=args.k,
        target_recall=args.target_recall,
        model_name=args.model,
        qrels_path=args.qrels,
        output_dir=args.output_dir,
        pool_output=args.pool_output,
        seed=args.seed,
        scale_multipliers=scale_multipliers,
    )


if __name__ == "__main__":
    main()
