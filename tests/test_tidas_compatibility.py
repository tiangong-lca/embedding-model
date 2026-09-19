"""Compatibility proof for the reviewed Python TIDAS SDK 0.2 runtime."""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
from tidas_sdk import create_process_from_json

from src.pre_process import convert_single_file
from src.utils import tidas_process_to_markdown


PROCESS_FIXTURE = {
    "processDataSet": {
        "processInformation": {
            "dataSetInformation": {
                "name": {
                    "baseName": [
                        {"@xml:lang": "en", "#text": "Electricity, solar, at plant"},
                        {"@xml:lang": "zh", "#text": "太阳能电力，工厂交付"},
                    ]
                },
                "common:UUID": "11111111-1111-4111-8111-111111111111",
                "classificationInformation": {
                    "common:classification": {
                        "common:class": [
                            {"@level": "0", "@classId": "D", "#text": "Electricity supply"},
                            {"@level": "1", "@classId": "35", "#text": "Power generation"},
                        ]
                    }
                },
            },
            "quantitativeReference": {
                "@type": "Reference flow(s)",
                "referenceToReferenceFlow": "1",
            },
        },
        "exchanges": {
            "exchange": [
                {
                    "@dataSetInternalID": "1",
                    "meanAmount": "1.0",
                    "exchangeDirection": "Output",
                    "referenceToFlowDataSet": {
                        "@refObjectId": "22222222-2222-4222-8222-222222222222",
                        "@type": "flow data set",
                        "@version": "01.00.000",
                        "@uri": "../flows/22222222-2222-4222-8222-222222222222.xml",
                        "common:shortDescription": [
                            {"@xml:lang": "en", "#text": "Electricity, medium voltage"}
                        ],
                    },
                }
            ]
        },
        "administrativeInformation": {
            "publicationAndOwnership": {"common:dataSetVersion": "01.00.000"}
        },
    }
}


class TidasCompatibilityTest(unittest.TestCase):
    def test_installed_sdk_is_the_reviewed_release(self) -> None:
        self.assertEqual(version("tidas-sdk"), "0.2.14")

    def test_adapter_delegates_to_canonical_sdk_markdown(self) -> None:
        process = create_process_from_json(json.dumps(PROCESS_FIXTURE))
        markdown = tidas_process_to_markdown(process, lang="en")

        self.assertEqual(markdown, process.to_markdown(lang="en"))
        self.assertIn("# Electricity, solar, at plant", markdown)
        self.assertIn("**Entity:** Process", markdown)
        self.assertIn("**Reference Flow:** Electricity, medium voltage", markdown)
        self.assertIn("**Classification:** Electricity supply > Power generation", markdown)

    def test_file_conversion_uses_the_same_canonical_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "process.json"
            output_path = Path(temp_dir) / "process.md"
            input_path.write_text(json.dumps(PROCESS_FIXTURE), encoding="utf-8")

            with redirect_stdout(StringIO()):
                result = convert_single_file(input_path, output_path=output_path, lang="en")

            self.assertEqual(result, output_path)
            self.assertIn("**Entity:** Process", output_path.read_text(encoding="utf-8"))

    def test_strict_validation_rejects_incomplete_processes(self) -> None:
        with self.assertRaises(ValidationError):
            create_process_from_json('{"processDataSet": {}}', validate=True)

    def test_pipeline_defaults_resolve_from_the_repository_root(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "01_generate_markdown.py"
        spec = importlib.util.spec_from_file_location("generate_markdown_pipeline", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda *args, **kwargs: None

        with patch.dict(sys.modules, {"tqdm": tqdm_module}):
            spec.loader.exec_module(module)

        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(module.DEFAULT_INPUT_DIR, repository_root / "data" / "tidas" / "processes")
        self.assertEqual(module.DEFAULT_OUTPUT_DIR, repository_root / "data" / "markdown")


if __name__ == "__main__":
    unittest.main()
