from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from view_alignment import (VIEW_NAMES, VIEW_AXES, RENDER_VERSION, canonical_frame,
                            cube_rotations, rotation_assignment, match_views,
                            align_view_files, source_signature, render_cache_valid,
                            render_standard_views)


class ViewAlignmentTests(unittest.TestCase):
    def make_views(self, root):
        paths = {}
        for i, name in enumerate(VIEW_NAMES):
            image = Image.new("RGB", (128, 128), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((18, 10, 108 - i * 4, 113 - i * 2), fill="#A7C1D1")
            draw.ellipse((27 + i * 5, 29, 45 + i * 5, 47), fill="white")
            draw.rectangle((18, 75 - i * 3, 37 + i * 2, 94), fill="white")
            path = root / f"{name}.png"
            image.save(path)
            paths[name] = str(path)
        return paths

    def test_rotations_are_proper_and_keep_opposites(self):
        rotations = cube_rotations()
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(r.flat) for r in rotations}), 24)
        for r in rotations:
            self.assertAlmostEqual(np.linalg.det(r), 1)
            pairs = rotation_assignment(r)
            self.assertEqual(len({c for _, c, _ in pairs}), 6)
            for q, c, k in pairs:
                np.testing.assert_array_equal(r @ VIEW_AXES[c][0], VIEW_AXES[q][0])
                self.assertIn(k, range(4))

    def test_every_cube_rotation_recovers_images_without_reflection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.make_views(root)
            for i, r in enumerate(cube_rotations()):
                folder = root / str(i)
                folder.mkdir()
                query = {}
                for q, c, k in rotation_assignment(r):
                    with Image.open(candidate[c]) as im:
                        path = folder / f"{q}.png"
                        Image.fromarray(np.rot90(np.asarray(im), k)).save(path)
                    query[q] = str(path)
                result = match_views(query, candidate)
                # Save/rotate/resize order introduces sub-pixel antialiasing,
                # so a correct rigid assignment is near one rather than exact.
                self.assertGreater(result["score"], 0.995)
                self.assertTrue(result["rigid_consistent"])
                np.testing.assert_array_equal(result["rotation_matrix"], r)
            aligned = align_view_files(query, candidate, root / "aligned")
            for n in VIEW_NAMES:
                with Image.open(query[n]) as a, Image.open(aligned["aligned_views"][n]) as b:
                    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            self.assertEqual(aligned, align_view_files(query, candidate, root / "aligned"))

    def test_query_back_can_use_candidate_top_rotated_90_degrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = self.make_views(root)
            target_rotation = next(
                rotation
                for rotation in cube_rotations()
                if ("back", "top", 1) in rotation_assignment(rotation)
            )
            query = {}
            for query_name, candidate_name, turns in rotation_assignment(target_rotation):
                with Image.open(candidate[candidate_name]) as image:
                    path = root / f"query_{query_name}.png"
                    Image.fromarray(np.rot90(np.asarray(image), turns)).save(path)
                query[query_name] = str(path)

            result = match_views(query, candidate, method="rigid24")
            pair = next(item for item in result["pairs"] if item["query_view"] == "back")
            self.assertEqual(pair["candidate_view"], "top")
            self.assertEqual(pair["rotation_degrees_ccw"], 90)
            self.assertGreater(result["score"], 0.995)

    def test_hungarian_detects_nonrigid_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self.make_views(Path(tmp))
            query = dict(candidate)
            query["front"], query["left"] = query["left"], query["front"]
            result = match_views(query, candidate)
            independent = match_views(query, candidate, "hungarian")
            self.assertGreater(independent["score"], result["score"])
            self.assertFalse(independent["rigid_consistent"])
            self.assertIsNone(independent["rotation_matrix"])

    def test_missing_and_blank_views_are_not_perfect_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            views = self.make_views(Path(tmp))
            with self.assertRaises(ValueError):
                match_views({}, views)
            Image.new("RGB", (128, 128), "white").save(views["top"])
            with self.assertRaises(ValueError):
                match_views(views, views)

    def test_render_cache_checks_source_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "part.stp"
            source.write_text("test")
            for n in VIEW_NAMES:
                (root / f"part_{n}.png").touch()
            self.assertFalse(render_cache_valid(source, root))
            (root / "render_manifest.json").write_text(json.dumps({
                "version": RENDER_VERSION, "source": source_signature(source)}))
            self.assertTrue(render_cache_valid(source, root))
            source.write_text("changed")
            self.assertFalse(render_cache_valid(source, root))

    def test_llm_labels_use_filename_instead_of_list_position(self):
        from stp_similarity import prepare_view_image_messages
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_views(Path(tmp))
            messages = prepare_view_image_messages([paths["back"], paths["top"], paths["front"]])
            labels = [m["text"] for m in messages if m["type"] == "text"]
            self.assertEqual(labels, ["\n【后视图】", "\n【俯视图】", "\n【主视图】"])

    def test_arbitrary_rotation_and_translation_preserve_canonical_box(self):
        import pyvista as pv
        from scipy.spatial.transform import Rotation
        mesh = pv.Box(bounds=(-2, 2, -4, 4, -0.7, 0.7)).triangulate()
        triangles = mesh.faces.reshape(-1, 4)[:, 1:]
        base, meta = canonical_frame(mesh.points, triangles)
        for seed in range(8):
            r = Rotation.random(random_state=seed).as_matrix()
            transformed = mesh.points @ r.T + [130, -245, 79]
            aligned, info = canonical_frame(transformed, triangles)
            np.testing.assert_allclose(np.ptp(aligned, axis=0), np.ptp(base, axis=0), atol=1e-6)
            np.testing.assert_allclose(aligned.min(axis=0) + aligned.max(axis=0), 0, atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(info["basis_columns"]), 1)

    def test_renderer_really_changes_camera_for_six_views(self):
        """Regression: a reused off-screen buffer once wrote one view 6 times."""
        import pyvista as pv

        # Deliberately asymmetric in all axes so distinct camera directions
        # cannot legitimately produce six identical silhouettes.
        body = pv.Box(bounds=(-3, 2, -2, 4, -0.8, 0.8)).triangulate()
        arm = pv.Box(bounds=(1, 4, 1.5, 3, -0.8, 2.6)).triangulate()
        mesh = body.merge(arm).triangulate()
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = render_standard_views(mesh, tmp, "asymmetric")
            hashes = []
            for path in paths:
                with Image.open(path) as image:
                    hashes.append(hash(image.convert("RGB").tobytes()))
            self.assertGreaterEqual(len(set(hashes)), 3)


if __name__ == "__main__":
    unittest.main()
