from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from stp_similarity import EmbeddingIndex, LLMReranker  # noqa: E402
from app.schemas import SearchRequest  # noqa: E402


def _candidates(count: int) -> list[dict]:
    return [
        {
            "filename": f"part-{index + 1}.stp",
            "path": f"/library/part-{index + 1}.stp",
            "similarity": 0.9 - index * 0.01,
            "text_similarity": 0.91 - index * 0.01,
            "geo_similarity": 0.82 - index * 0.01,
            "visual_similarity": 0.73 - index * 0.01,
            "fusion_score": 0.88 - index * 0.01,
            "hybrid_similarity": 0.95 - index * 0.01,
            "geometric_similarity": {"overall": 0.8 - index * 0.01},
            "recall_sources": ["text", "geometric", "visual"],
            "recall_count": 3,
        }
        for index in range(count)
    ]


class RerankCompletionTests(unittest.TestCase):
    def test_binary_file_is_rejected_instead_of_producing_zero_features(self):
        import tempfile
        from stp_similarity import parse_stp_deep
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encrypted.stp"
            path.write_bytes(b"\x93\xa1\x00not-a-step-file")
            with self.assertRaisesRegex(ValueError, "ISO-10303-21"):
                parse_stp_deep(str(path))

    def test_three_way_recall_skips_invalid_index_candidate(self):
        import tempfile
        import numpy as np

        class FakeCollection:
            @staticmethod
            def count():
                return 1

            @staticmethod
            def query(**_kwargs):
                return {
                    "ids": [["part_0001"]],
                    "metadatas": [[{"path": invalid_path}]],
                    "distances": [[0.1]],
                }

        with tempfile.TemporaryDirectory() as tmp:
            invalid_path = str(Path(tmp) / "encrypted.stp")
            Path(invalid_path).write_bytes(b"\x17\xda\x5f\xa0not-a-step-file")

            index = object.__new__(EmbeddingIndex)
            index.collection = FakeCollection()
            index.geo_index = None
            index.visual_index = None
            index._info_cache = {}
            index._invalid_info_cache = {}

            candidates, *_ = index.three_way_recall(
                query_path=str(Path(tmp) / "query.stp"),
                query_text="valid query description",
                query_info={"mfg_features": {}},
                query_geo_vector=np.zeros(64, dtype=np.float32),
            )

            self.assertEqual(candidates, [])
            self.assertIn(invalid_path, index._invalid_info_cache)

    def test_search_request_defaults_to_ten_results(self):
        request = SearchRequest(file_id="query-file")
        self.assertEqual(request.effective_result_limit, 10)

    def test_search_request_defaults_to_three_way_recall(self):
        request = SearchRequest(file_id="query-file")
        self.assertTrue(request.use_three_way)

    def test_result_limit_overrides_legacy_final_top(self):
        request = SearchRequest(
            file_id="query-file", result_limit=10, final_top=5
        )
        self.assertEqual(request.effective_result_limit, 10)
        self.assertEqual(request.final_top, 10)

    def test_multiple_libraries_are_normalized(self):
        request = SearchRequest(
            file_id="query-file",
            library_id="legacy-library",
            library_ids=["u-library", "l-library", "u-library"],
        )
        self.assertEqual(request.effective_library_ids, ["u-library", "l-library"])
        self.assertIsNone(request.library_id)

    def test_partial_llm_rankings_are_completed_to_requested_count(self):
        candidates = _candidates(11)
        llm_rankings = [
            {
                "candidate_number": index,
                "filename": f"model-output-{index}.stp",
                "similarity_score": 0.99 - index * 0.01,
                "reason": "LLM排名",
            }
            for index in (3, 1, 5, 2, 4)
        ]

        completed = LLMReranker._complete_rankings(
            llm_rankings, candidates, top_k=11
        )

        self.assertEqual(len(completed), 11)
        self.assertEqual(
            [item["candidate_number"] for item in completed[:5]],
            [3, 1, 5, 2, 4],
        )
        self.assertEqual(len({item["candidate_number"] for item in completed}), 11)
        self.assertEqual(completed[0]["filename"], "part-3.stp")
        self.assertAlmostEqual(completed[0]["visual_similarity"], 0.71)
        self.assertAlmostEqual(completed[0]["text_similarity"], 0.89)
        self.assertAlmostEqual(completed[0]["geo_similarity"], 0.80)
        self.assertEqual(completed[0]["recall_sources"], ["text", "geometric", "visual"])
        self.assertEqual(completed[-1]["candidate_number"], 11)
        self.assertIn("补齐", completed[-1]["reason"])

    def test_invalid_and_duplicate_llm_items_do_not_reduce_count(self):
        candidates = _candidates(4)
        llm_rankings = [
            {"candidate_number": 1, "reason": "first"},
            {"candidate_number": 1, "reason": "duplicate"},
            {"candidate_number": 99, "reason": "invalid"},
            {"filename": "part-3.stp", "reason": "matched by filename"},
        ]

        completed = LLMReranker._complete_rankings(
            llm_rankings, candidates, top_k=4
        )

        self.assertEqual(len(completed), 4)
        self.assertEqual(
            {item["candidate_number"] for item in completed},
            {1, 2, 3, 4},
        )


if __name__ == "__main__":
    unittest.main()
