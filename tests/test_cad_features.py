from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.cad_features import (  # noqa: E402
    FEATURE_RECOGNITION_VERSION,
    LEGACY_FEATURE_RECOGNITION_VERSION,
    recognize_features_v11,
)
from stp_similarity import extract_manufacturing_features_standalone  # noqa: E402


def _entity(entity_type: str, args: str) -> dict[str, str]:
    return {"type": entity_type, "args": args}


def _single_hole_entities(*, blind: bool = False) -> dict[str, dict[str, str]]:
    entities = {
        "#1": _entity("CARTESIAN_POINT", "'',(0.,0.,0.)"),
        "#2": _entity("DIRECTION", "'',(0.,0.,1.)"),
        "#3": _entity("DIRECTION", "'',(1.,0.,0.)"),
        "#4": _entity("AXIS2_PLACEMENT_3D", "'',#1,#2,#3"),
        "#5": _entity("CYLINDRICAL_SURFACE", "'',#4,5."),
        "#6": _entity("CARTESIAN_POINT", "'',(5.,0.,0.)"),
        "#7": _entity("CARTESIAN_POINT", "'',(5.,0.,10.)"),
        "#8": _entity("VERTEX_POINT", "'',#6"),
        "#9": _entity("VERTEX_POINT", "'',#7"),
        "#10": _entity("CIRCLE", "'',#4,5."),
        "#11": _entity("CIRCLE", "'',#4,5."),
        "#12": _entity("EDGE_CURVE", "'',#8,#8,#10,.T."),
        "#13": _entity("EDGE_CURVE", "'',#9,#9,#11,.T."),
        "#14": _entity("ORIENTED_EDGE", "'',*,*,#12,.T."),
        "#15": _entity("ORIENTED_EDGE", "'',*,*,#13,.T."),
        "#16": _entity("EDGE_LOOP", "'',(#14,#15)"),
        "#17": _entity("FACE_OUTER_BOUND", "'',#16,.T."),
        "#18": _entity("ADVANCED_FACE", "'',(#17),#5,.F."),
        # Top exterior face: the shared circle is an inner boundary/opening.
        "#20": _entity("PLANE", "'',#4"),
        "#21": _entity("EDGE_LOOP", "'',(#14)"),
        "#22": _entity("FACE_BOUND", "'',#21,.T."),
        "#23": _entity("ADVANCED_FACE", "'',(#22),#20,.T."),
        # Bottom face: inner boundary means another opening; outer means a cap.
        "#30": _entity("CARTESIAN_POINT", "'',(0.,0.,10.)"),
        "#31": _entity("AXIS2_PLACEMENT_3D", "'',#30,#2,#3"),
        "#32": _entity("PLANE", "'',#31"),
        "#33": _entity("EDGE_LOOP", "'',(#15)"),
        "#34": _entity("FACE_OUTER_BOUND" if blind else "FACE_BOUND", "'',#33,.T."),
        "#35": _entity("ADVANCED_FACE", "'',(#34),#32,.T."),
    }
    # External cylinders (bosses/rounded exterior) must not become holes.
    for index in range(40, 53):
        entities[f"#{index}"] = _entity("ADVANCED_FACE", "'',(),#5,.T.")
    return entities


class CADFeatureRecognitionTests(unittest.TestCase):
    def test_internal_version_moves_from_1_0_to_1_1(self):
        self.assertEqual(LEGACY_FEATURE_RECOGNITION_VERSION, "1.0")
        self.assertEqual(FEATURE_RECOGNITION_VERSION, "1.1")

    def test_one_hole_is_not_counted_as_all_cylindrical_faces(self):
        result = recognize_features_v11(
            _single_hole_entities(),
            {"bbox_diagonal": 40.0, "bbox_dims": [10.0, 20.0, 30.0]},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["through_holes"], 1)
        self.assertEqual(result["blind_holes"], 0)

    def test_outer_bottom_boundary_classifies_blind_hole(self):
        result = recognize_features_v11(
            _single_hole_entities(blind=True),
            {"bbox_diagonal": 40.0, "bbox_dims": [10.0, 20.0, 30.0]},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["through_holes"], 0)
        self.assertEqual(result["blind_holes"], 1)

    def test_existing_manufacturing_contract_uses_v11_without_new_api_fields(self):
        result = extract_manufacturing_features_standalone(
            _single_hole_entities(),
            {
                "bbox_diagonal": 40.0,
                "bbox_dims": [10.0, 20.0, 30.0],
                "cylinder_radii": [5.0],
                "face_types": {"plane": 2, "cylinder": 14},
                "num_faces": 16,
                "edge_types": {"circle": 2, "line": 0, "bspline": 0},
                "num_edges": 2,
            },
        )
        self.assertEqual(result["through_holes"], 1)
        self.assertEqual(result["blind_holes"], 0)
        self.assertEqual(len(result["mfg_feature_vector"]), 32)
        self.assertNotIn("feature_recognition_version", result)


if __name__ == "__main__":
    unittest.main()
