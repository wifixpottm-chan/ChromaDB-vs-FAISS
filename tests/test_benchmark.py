import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    _export_judgment_pool,
    _measure_paired_searches,
    _validate_scale_multipliers,
    _write_run_artifacts,
    labeled_metrics,
    load_relevance_judgments,
    mean_recall_at_k,
    ndcg_at_k,
    select_matched_configurations,
)


class BenchmarkMetricsTests(unittest.TestCase):
    def test_mean_recall_at_k_compares_to_exact_neighbor_sets(self):
        exact = {"first": ["a", "b"], "second": ["c", "d"]}
        approximate = {"first": ["a", "x"], "second": ["d", "c"]}

        self.assertEqual(mean_recall_at_k(approximate, exact, 2), 0.75)

    def test_labeled_metrics_use_curated_relevant_ids(self):
        results = {"query": ["other", "relevant", "last"]}
        relevance = {"query": {"relevant", "missing"}}

        self.assertEqual(labeled_metrics(results, relevance, 3), (0.5, 0.5))

    def test_ndcg_rewards_relevant_items_at_higher_ranks(self):
        judgments = {"query": {"relevant": 1, "other-relevant": 1, "not-relevant": 0}}

        best = ndcg_at_k(
            {"query": ["relevant", "other-relevant", "not-relevant"]}, judgments, 3
        )
        worse = ndcg_at_k(
            {"query": ["not-relevant", "relevant", "other-relevant"]}, judgments, 3
        )

        self.assertEqual(best, 1.0)
        self.assertLess(worse, best)

    def test_loads_relevance_csv_and_rejects_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            qrels_path = Path(directory) / "qrels.csv"
            qrels_path.write_text(
                "query,episode_id,relevance\nFind episode,42,1\n",
                encoding="utf-8",
            )

            queries, judgments = load_relevance_judgments(qrels_path, {"42"})

            self.assertEqual(queries, ["Find episode"])
            self.assertEqual(judgments, {"Find episode": {"42": 1}})
            with self.assertRaisesRegex(ValueError, "unknown episode_id"):
                load_relevance_judgments(qrels_path, {"1"})

    def test_paired_search_measurement_warms_and_repeats_each_backend(self):
        calls = {"first": 0, "second": 0}
        searchers = {
            name: lambda vector, key=name: (calls.__setitem__(key, calls[key] + 1) or [vector])
            for name in calls
        }

        results, timings = _measure_paired_searches(
            searchers,
            [1, 2],
            ["q1", "q2"],
            repeats=2,
            k=1,
            seed=7,
        )

        self.assertEqual(calls, {"first": 6, "second": 6})
        self.assertEqual(len(timings["first"]), 4)
        self.assertEqual(results["second"], {"q1": [1], "q2": [2]})

    def test_judgment_pool_includes_seed_labels_even_if_not_retrieved(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pool.csv"
            _export_judgment_pool(
                output_path,
                ["query"],
                {"query": ["retrieved"]},
                [{"backend": "FAISS HNSW", "setting": "ef=8", "results": {"query": ["ann"]}}],
                {"query": {"gold": 3}},
            )

            with output_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(
                {(row["episode_id"], row["relevance"]) for row in rows},
                {("gold", "3"), ("retrieved", ""), ("ann", "")},
            )

    def test_scale_multipliers_must_be_nonempty_and_positive(self):
        _validate_scale_multipliers([1, 2, 5])
        with self.assertRaises(ValueError):
            _validate_scale_multipliers([])
        with self.assertRaises(ValueError):
            _validate_scale_multipliers([0, 2])

    def test_run_artifacts_include_run_metadata_and_nested_operations(self):
        summary = {
            "created_at_utc": "2026-01-01T00:00:00Z",
            "model": "test-model",
            "k": 5,
            "repeats": 2,
            "versions": {
                "python": "3.14",
                "platform": "test",
                "numpy": "1",
                "faiss-cpu": "2",
                "chromadb": "3",
                "sentence-transformers": "4",
            },
        }
        configurations = [
            {
                "dataset_size": 10,
                "dataset_type": "real",
                "record_type": "search_configuration",
                "backend": "FAISS Flat",
                "setting": "exact",
                "recall": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 2.0,
            }
        ]
        operations = {"metadata_filter_comparison": {"latency_ms": 1.5}}

        with tempfile.TemporaryDirectory() as directory:
            json_path, csv_path, operations_path = _write_run_artifacts(
                Path(directory), summary, configurations, operations, []
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as file:
                search_rows = list(csv.DictReader(file))
            with operations_path.open(encoding="utf-8", newline="") as file:
                operation_rows = list(csv.DictReader(file))

        self.assertEqual(payload["summary"]["versions"]["faiss-cpu"], "2")
        self.assertEqual(search_rows[0]["model"], "test-model")
        self.assertEqual(operation_rows[0]["operation"], "latency_ms")

    def test_selects_fastest_config_at_common_recall_target(self):
        sweeps = {
            "FAISS": [
                {"setting": "low", "recall": 0.8, "p50_ms": 1.0},
                {"setting": "high", "recall": 1.0, "p50_ms": 2.0},
            ],
            "Chroma": [
                {"setting": "low", "recall": 0.9, "p50_ms": 2.0},
                {"setting": "high", "recall": 0.98, "p50_ms": 3.0},
            ],
        }

        matched_recall, selected = select_matched_configurations(sweeps, 0.95)

        self.assertEqual(matched_recall, 0.95)
        self.assertEqual(selected["FAISS"]["setting"], "high")
        self.assertEqual(selected["Chroma"]["setting"], "high")

    def test_lowers_common_target_when_requested_recall_is_unavailable(self):
        sweeps = {
            "FAISS": [{"setting": "best", "recall": 1.0, "p50_ms": 1.0}],
            "Chroma": [{"setting": "best", "recall": 0.8, "p50_ms": 2.0}],
        }

        matched_recall, selected = select_matched_configurations(sweeps, 0.95)

        self.assertEqual(matched_recall, 0.8)
        self.assertEqual(selected["Chroma"]["setting"], "best")


if __name__ == "__main__":
    unittest.main()
