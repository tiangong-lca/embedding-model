"""Bounded legacy consumer snapshots for the deliberate TIDAS 0.3 adoption."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime
import importlib.util
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from pydantic import ValidationError
from tidas_sdk import create_flow_from_json, create_process_from_json
from src.pre_process import convert_single_file
from src.utils import tidas_process_to_markdown

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((Path(__file__).parent / "fixtures/tidas-consumers.json").read_text(encoding="utf-8"))
FIXED = datetime.fromisoformat(FIXTURES["fixedTimestamp"])


def load_pipeline():
    spec = importlib.util.spec_from_file_location("consumer_pipeline", ROOT / "scripts/pipeline/01_generate_markdown.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TidasConsumerTest(unittest.TestCase):
    def test_entity_and_pipeline_language_snapshots(self):
        """60 entity outcomes and 60 real file conversions, including fallback."""
        pipeline = load_pipeline()
        with tempfile.TemporaryDirectory() as temporary, \
             patch("tidas_sdk.entities.process.default_timestamp", return_value=FIXED), \
             patch("tidas_sdk.entities.flow.default_timestamp", return_value=FIXED), \
             patch.object(pipeline, "default_timestamp", return_value=FIXED):
            root = Path(temporary)
            for case in FIXTURES["cases"]:
                factory = create_process_from_json if case["kind"] == "process" else create_flow_from_json
                for lang, expected in case["expectedMarkdown"].items():
                    with self.subTest(kind=case["kind"], case=case["case"], lang=lang):
                        raw = json.dumps(case["input"])
                        if case.get("directError"):
                            # Pre-existing bare Flow behavior; the actual pipeline repairs the timestamp.
                            self.assertEqual(case["directError"], "AttributeError")
                            with self.assertRaises(AttributeError):
                                factory(raw)
                        else:
                            entity = factory(raw)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                before = entity.to_json(mode="json")
                            self.assertEqual(before, case["expectedParsed"])
                            self.assertEqual(entity.to_markdown(lang=lang), expected)
                            self.assertEqual(entity.to_markdown(lang=lang), expected)
                            if case["kind"] == "process":
                                self.assertEqual(tidas_process_to_markdown(entity, lang), expected)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                self.assertEqual(entity.to_json(mode="json"), before)
                        input_path = root / f"{case['kind']}-{case['case']}-{lang}.json"
                        input_path.write_text(raw)
                        output = pipeline.convert_json_to_markdown(input_path, root / "output", lang)
                        self.assertEqual(output.read_text(encoding="utf-8"), expected)
                        # This intentionally records the existing processs/ spelling.
                        self.assertEqual(output.parent.name, "processs" if case["kind"] == "process" else "flows")

    def test_timestamp_sanitization(self):
        pipeline = load_pipeline()
        fixtures = [
            ("process", {}),
            ("flow", {"administrativeInformation": []}),
            ("flow", {"administrativeInformation": {"dataEntryBy": {"common:timeStamp": ""}}}),
            ("process", {"administrativeInformation": {"dataEntryBy": {"common:timeStamp": "2020-01-01T01:02:03Z"}}}),
        ]
        with patch.object(pipeline, "default_timestamp", return_value=FIXED):
            for kind, dataset in fixtures:
                with self.subTest(kind=kind, dataset=dataset):
                    entity, raw = pipeline._sanitize_payload({kind + "DataSet": deepcopy(dataset)})
                    self.assertEqual(entity, kind)
                    actual = json.loads(raw)[kind + "DataSet"]["administrativeInformation"]["dataEntryBy"]["common:timeStamp"]
                    previous = dataset.get("administrativeInformation")
                    timestamp = previous.get("dataEntryBy", {}).get("common:timeStamp") if isinstance(previous, dict) else None
                    self.assertEqual(actual, timestamp or FIXED.isoformat())

    def test_strict_incomplete_process_and_flow_still_reject(self):
        for kind, factory in [("process", create_process_from_json), ("flow", create_flow_from_json)]:
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                factory(json.dumps({kind + "DataSet": {}}), validate=True)

    def test_batch_order_resume_and_preprocessing(self):
        pipeline = load_pipeline()
        fixtures = {case["kind"]: case for case in FIXTURES["cases"] if case["case"] == "baseline"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, folder in [("flow", "flows"), ("process", "processes")]:
                (root / "input" / folder).mkdir(parents=True)
                (root / "input" / folder / "b.json").write_text(json.dumps(fixtures[kind]["input"]))
            (root / "input/processes/a.json").write_text(json.dumps(fixtures["process"]["input"]))
            (root / "input/processes/z-invalid.json").write_text("{broken")
            self.assertEqual(
                [path.relative_to(root / "input").as_posix() for path in pipeline.iter_json_files(root / "input")],
                ["flows/b.json", "processes/a.json", "processes/b.json", "processes/z-invalid.json"],
            )
            # Do not install signal handlers on the unittest process.
            with patch.object(pipeline.signal, "signal"), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(pipeline.generate_markdown_files(root / "input", root / "batch"), {"success": 3, "failed": 1})
                self.assertEqual(pipeline.generate_markdown_files(root / "input", root / "resume", start_index=1), {"success": 2, "failed": 1})
                output = convert_single_file(root / "input/processes/a.json", root / "single.md", "zh")
            self.assertEqual(output.read_text(encoding="utf-8"), fixtures["process"]["expectedMarkdown"]["zh"])
            self.assertEqual((root / "batch/flows/b.md").read_text(encoding="utf-8"), fixtures["flow"]["expectedMarkdown"]["en"])
            self.assertEqual((root / "batch/processs/a.md").read_text(encoding="utf-8"), fixtures["process"]["expectedMarkdown"]["en"])
