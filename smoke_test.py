"""Offline API smoke test; no model API or CAD renderer is required."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stpnew-test-") as temp_dir:
        os.environ["RUNTIME_DIR"] = temp_dir
        os.environ.pop("API_KEY", None)
        os.environ["ALLOW_LOCAL_PATHS"] = "false"

        from fastapi.testclient import TestClient
        from app.main import app
        from app.main_impl import engines, renders, store

        sample = b"""ISO-10303-21;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,1.,1.));
ENDSEC;
END-ISO-10303-21;
"""
        with TestClient(app, raise_server_exceptions=False) as client:
            root = client.get("/")
            assert root.status_code == 200
            assert root.json()["port"] == 8001

            status = client.get("/api/status")
            assert status.status_code == 200
            assert status.json()["port"] == 8001
            assert status.json()["library_count"] == 1

            created_library = client.post(
                "/api/libraries",
                json={"name": "测试零件库", "description": "offline smoke"},
            )
            assert created_library.status_code == 201, created_library.text
            library_id = created_library.json()["library_id"]
            assert renders._cache_root(library_id).resolve() == engines._view_path(
                library_id
            ).resolve()

            original_build_index = engines.build_index
            original_render = renders.render
            engines.build_index = lambda *args, **kwargs: {
                "text": 1, "geometric": 1, "visual": 1
            }
            renders.render = lambda *args, **kwargs: (
                {
                    "front": "/media/renders/front.png",
                    "back": "/media/renders/back.png",
                    "top": "/media/renders/top.png",
                    "bottom": "/media/renders/bottom.png",
                    "left": "/media/renders/left.png",
                    "right": "/media/renders/right.png",
                },
                None,
                False,
            )
            library_upload = client.post(
                "/api/files/upload-batch",
                files=[
                    ("files", ("library_part.stp", sample, "application/octet-stream"))
                ],
                data={
                    "category": "library",
                    "library_id": library_id,
                    "auto_index": "true",
                },
            )
            engines.build_index = original_build_index
            renders.render = original_render
            assert library_upload.status_code == 200, library_upload.text
            upload_body = library_upload.json()
            assert upload_body["library_id"] == library_id
            assert upload_body["index"]["status"] == "ready"
            assert upload_body["index"]["text_count"] == 1
            assert upload_body["index_error"] is None
            assert upload_body["renders"][0]["status"] == "ready"
            assert upload_body["renders"][0]["views"]["front"].endswith("front.png")
            assert upload_body["processing"]["rendered_count"] == 1
            assert upload_body["processing"]["complete"] is True
            assert upload_body["files"][0]["metadata"]["total_entities"] == 2
            library_part_id = upload_body["files"][0]["file_id"]

            parts = client.get(
                "/api/parts", params={"library_id": library_id, "limit": 20}
            )
            assert parts.status_code == 200, parts.text
            assert parts.json()["total"] == 1
            assert parts.json()["items"][0]["library_id"] == library_id

            library_detail = client.get(f"/api/libraries/{library_id}")
            assert library_detail.status_code == 200
            assert library_detail.json()["part_count"] == 1

            second_library = client.post(
                "/api/libraries",
                json={"name": "第二测试零件库", "description": "multi-library"},
            )
            assert second_library.status_code == 201, second_library.text
            second_library_id = second_library.json()["library_id"]
            store.set_library_index_state(
                second_library_id,
                "ready",
                text_count=1,
                geometric_count=1,
                visual_count=1,
            )

            upload = client.post(
                "/api/files/upload",
                files={"file": ("sample.stp", sample, "application/octet-stream")},
                data={"category": "query"},
            )
            assert upload.status_code == 200, upload.text
            file_id = upload.json()["file"]["file_id"]

            # Exercise the public pairing contract with rendering isolated.
            original_match = renders.match
            def fake_match(query, candidate, *args, **kwargs):
                assert query == store.resolve_file(file_id, None, {".stp"})
                from view_alignment import VIEW_NAMES
                urls = {n: f"/media/renders/aligned_{n}.png" for n in VIEW_NAMES}
                return {"status": "ready", "method": "rigid24", "score": 0.98,
                        "hungarian_score": 0.99, "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]],
                        "rigid_consistent": True, "score_gap": 0.05, "ambiguous": False,
                        "pairs": [{"query_view": n, "candidate_view": n,
                                   "rotation_degrees_ccw": 0, "score": 0.98} for n in VIEW_NAMES],
                        "query_views": urls, "candidate_views": urls,
                        "aligned_views": urls, "warning": ""}
            renders.match = fake_match
            matched = client.post("/api/render/match", json={
                "query_file_id": file_id, "candidate_file_id": library_part_id})
            assert matched.status_code == 200, matched.text
            assert len(matched.json()["pairs"]) == 6
            assert matched.json()["aligned_views"]["front"].endswith("aligned_front.png")
            invalid_match = client.post("/api/render/match", json={
                "query_file_id": file_id, "candidate_file_id": library_part_id, "method": "mirror"})
            assert invalid_match.status_code == 422, invalid_match.text
            missing_match = client.post("/api/render/match", json={
                "query_file_id": "missing", "candidate_file_id": library_part_id})
            assert missing_match.status_code == 404, missing_match.text

            original_search = engines.search
            def fake_search(selected_library_id, *args, **kwargs):
                if selected_library_id == second_library_id:
                    return [{
                        "filename": "second_library_best.stp",
                        "filepath": str(Path(temp_dir) / "second_library_best.stp"),
                        "embedding_similarity": 0.96,
                        "similarity_score": 0.96,
                        "hybrid_similarity": 0.99,
                    }]
                return [
                    {
                        "filename": "low_score.stp",
                        "filepath": str(Path(temp_dir) / "low_score.stp"),
                        "embedding_similarity": 0.72,
                        "similarity_score": 0.98,
                        "hybrid_similarity": 0.80,
                    },
                    {
                        "filename": "high_score.stp",
                        "filepath": str(Path(temp_dir) / "high_score.stp"),
                        "embedding_similarity": 0.97,
                        "similarity_score": 0.70,
                        "hybrid_similarity": 0.90,
                    },
                    {
                        "filename": "middle_score.stp",
                        "filepath": store.record(library_part_id)["path"],
                        "embedding_similarity": 0.91,
                        "similarity_score": 0.85,
                        "hybrid_similarity": 0.95,
                    },
                ]
            engines.search = fake_search
            searched = client.post(
                "/api/search",
                json={
                    "file_id": file_id,
                    "library_id": library_id,
                    "result_limit": 2,
                    "use_llm_rerank": False,
                },
            )
            assert searched.status_code == 200, searched.text
            assert searched.json()["library_id"] == library_id
            assert searched.json()["library_ids"] == [library_id]
            assert searched.json()["result_limit"] == 2
            assert searched.json()["total_results"] == 2
            assert [item["filename"] for item in searched.json()["results"]] == [
                "middle_score.stp",
                "high_score.stp",
            ]
            assert [item["rank"] for item in searched.json()["results"]] == [1, 2]
            assert [item["composite_score"] for item in searched.json()["results"]] == [
                0.95,
                0.90,
            ]
            assert searched.json()["results"][0]["file_id"] == library_part_id
            paired_search = client.post("/api/search", json={
                "file_id": file_id, "library_id": library_id, "result_limit": 2,
                "use_llm_rerank": False, "include_view_alignment": True})
            assert paired_search.status_code == 200, paired_search.text
            assert paired_search.json()["results"][0]["view_alignment"]["rigid_consistent"]
            def failed_match(*args, **kwargs):
                raise ValueError("test render failed")
            renders.match = failed_match
            failed_pair_search = client.post("/api/search", json={
                "file_id": file_id, "library_id": library_id, "result_limit": 2,
                "use_llm_rerank": False, "include_view_alignment": True})
            assert failed_pair_search.status_code == 200, failed_pair_search.text
            assert failed_pair_search.json()["total_results"] == 2
            assert failed_pair_search.json()["results"][0]["view_alignment_error"] == "test render failed"
            renders.match = original_match

            multi_library_search = client.post(
                "/api/search",
                json={
                    "file_id": file_id,
                    "library_ids": [library_id, second_library_id],
                    "result_limit": 3,
                    "use_llm_rerank": False,
                },
            )
            engines.search = original_search
            assert multi_library_search.status_code == 200, multi_library_search.text
            multi_body = multi_library_search.json()
            assert multi_body["library_id"] is None
            assert multi_body["library_name"] is None
            assert multi_body["library_ids"] == [library_id, second_library_id]
            assert multi_body["total_results"] == 3
            assert [item["filename"] for item in multi_body["results"]] == [
                "second_library_best.stp",
                "middle_score.stp",
                "high_score.stp",
            ]
            assert [item["library_id"] for item in multi_body["results"]] == [
                second_library_id,
                library_id,
                library_id,
            ]
            inspection = client.get(f"/api/files/{file_id}/inspect")
            assert inspection.status_code == 200, inspection.text
            assert inspection.json()["metadata"]["total_entities"] == 2

            download = client.get(f"/api/files/{file_id}/download")
            assert download.status_code == 200
            assert download.content == sample

            invalid_reference = client.post(
                "/api/search",
                json={"file_id": file_id, "file_path": "sample.stp"},
            )
            assert invalid_reference.status_code == 422

            local_path = client.post(
                "/api/render", json={"file_path": str(Path(temp_dir) / "sample.stp")}
            )
            assert local_path.status_code == 403

            # The fixture is intentionally a tiny parser sample rather than a
            # complete CAD model. Mock rendering so this offline smoke test is
            # stable both with and without pythonocc-core installed.
            original_render = renders.render
            renders.render = lambda *args, **kwargs: (
                {
                    "front": "/media/renders/front.png",
                    "back": "/media/renders/back.png",
                    "top": "/media/renders/top.png",
                    "bottom": "/media/renders/bottom.png",
                    "left": "/media/renders/left.png",
                    "right": "/media/renders/right.png",
                },
                None,
                False,
            )
            try:
                render = client.post("/api/render", json={"file_id": file_id})
            finally:
                renders.render = original_render
            assert render.status_code == 200, render.text
            assert render.json()["views"]["front"].endswith("front.png")

            openapi = client.get("/openapi.json")
            assert openapi.status_code == 200
            assert "/api/files/upload" in openapi.json()["paths"]
            assert "/api/render" in openapi.json()["paths"]
            assert "/api/libraries" in openapi.json()["paths"]
            assert "/api/parts" in openapi.json()["paths"]
            assert "delete" in openapi.json()["paths"]["/api/parts/{file_id}"]
            assert "/api/libraries/{library_id}/build-index" in openapi.json()["paths"]

            # Deletion synchronously removes the catalog record, source file,
            # render cache and three-way index entry. Index internals are mocked
            # here so this smoke test stays completely offline.
            library_part_path = Path(store.record(library_part_id)["path"])
            render_cache = renders._cache_root(library_id) / library_part_path.stem
            render_cache.mkdir(parents=True, exist_ok=True)
            (render_cache / "front.png").write_bytes(b"png")
            original_remove_part = engines.remove_part_from_index
            engines.remove_part_from_index = lambda *args, **kwargs: (
                {"text": True, "geometric": True, "visual": True},
                {"text": 0, "geometric": 0, "visual": 0},
            )
            deleted = client.delete(f"/api/parts/{library_part_id}")
            engines.remove_part_from_index = original_remove_part
            assert deleted.status_code == 200, deleted.text
            delete_body = deleted.json()
            assert delete_body["file_id"] == library_part_id
            assert delete_body["file_deleted"] is True
            assert delete_body["render_deleted"] is True
            assert delete_body["index_removed"]["text"] is True
            assert delete_body["remaining_parts"] == 0
            assert delete_body["index"]["status"] == "not_built"
            assert not library_part_path.exists()
            assert not render_cache.exists()
            assert client.get(
                "/api/parts", params={"library_id": library_id}
            ).json()["total"] == 0
            assert client.delete(f"/api/parts/{library_part_id}").status_code == 404

    print("STP New smoke test: PASS")


if __name__ == "__main__":
    main()
