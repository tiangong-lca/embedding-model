"""Offline proof for the supporting clients actually used by this repository."""
from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import dotenv
import httpx
import httpx2
import numpy as np
import openai
from PIL import Image
from postgrest.exceptions import APIError
from supabase import ClientOptions, create_client

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def module_override(name, module):
    """Replace one module without evicting lazily imported library modules."""
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.connect = patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected real network"))
        self.create_connection = patch.object(socket, "create_connection", side_effect=AssertionError("Unexpected real network"))
        self.connect.start()
        self.create_connection.start()
        self.addCleanup(self.connect.stop)
        self.addCleanup(self.create_connection.stop)


class DotenvConsumerTest(OfflineTest):
    def test_real_config_uses_owned_values_and_preserves_environment_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text("DASHSCOPE_API_KEY=dummy-dashscope\nSILICONFLOW_API_KEY=dummy-siliconflow\nBASE=file\nEXPANDED=${BASE}/suffix\nEMPTY= # comment\n")
            real_load = dotenv.load_dotenv
            with patch.dict(os.environ, {"BASE": "environment"}, clear=True), \
                 patch.object(dotenv, "find_dotenv", return_value=str(path)), \
                 patch.object(dotenv, "load_dotenv", side_effect=lambda: real_load(path)):
                config = load_module("consumer_config", ROOT / "src/config/model.py")
                self.assertEqual(config.DASHSCOPE_API_KEY, "dummy-dashscope")
                self.assertEqual(config.SILICONFLOW_API_KEY, "dummy-siliconflow")
                self.assertEqual(os.environ["EXPANDED"], "environment/suffix")
                # Reviewed 1.2.4 correction; older 1.2.3 returned literal '# comment'.
                self.assertEqual(os.environ["EMPTY"], "")
                self.assertEqual(dotenv.get_key(path, "EMPTY"), "")
                self.assertEqual(dotenv.dotenv_values(path)["EXPANDED"], "file/suffix")


class OpenAIConsumerTest(OfflineTest):
    def test_real_query_requests_responses_and_errors(self):
        requests = []
        fail = False

        def handler(request):
            self.assertEqual(request.url.host, "consumer.invalid")
            body = json.loads(request.content)
            requests.append((request.method, request.url.path, body))
            if fail:
                return httpx2.Response(400, request=request, json={"error": {"message": "dummy request rejected", "type": "invalid_request_error"}})
            return httpx2.Response(200, request=request, json={
                "id": "chatcmpl-dummy", "object": "chat.completion", "created": 1, "model": "qwen-plus",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"queries":["a","b","c","d","e"]}'}, "finish_reason": "stop"}],
            })

        config = types.ModuleType("src.config.model")
        config.DASHSCOPE_API_KEY = "dummy"
        config.DASHSCOPE_API_URL = "https://consumer.invalid/v1"
        with openai.OpenAI(api_key="dummy", base_url=config.DASHSCOPE_API_URL, max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)) as client:
            with module_override("src.config.model", config), patch.object(openai, "OpenAI", return_value=client):
                consumer = load_module("consumer_queries", ROOT / "src/utils/llm_generate_query.py")
                for kind in ["process", "flow"]:
                    self.assertEqual(json.loads(consumer.generate_query("# Electricity", kind))["queries"], ["a", "b", "c", "d", "e"])
                fail = True
                with self.assertRaises(openai.BadRequestError) as error:
                    consumer.generate_query("bad", "flow")
                self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(len(requests), 3)
        for method, path, body in requests:
            self.assertEqual((method, path, body["model"]), ("POST", "/v1/chat/completions", "qwen-plus"))
            schema = body["response_format"]["json_schema"]
            self.assertEqual(schema["name"], "search_queries")
            self.assertIs(schema["strict"], True)
            self.assertEqual(schema["schema"]["properties"]["queries"]["minItems"], 5)
        self.assertIn("Material/Process", requests[0][2]["messages"][0]["content"])
        self.assertIn("Chemical/Material identity", requests[1][2]["messages"][0]["content"])

    def test_actual_classifier_function_without_importing_gpu_dependencies(self):
        path = ROOT / "scripts/tools/mine_hard_negatives.py"
        function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "classify_candidate")
        scope = {"OpenAI": openai.OpenAI, "json": json}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, request=request, json={
                "id": "chatcmpl-dummy", "object": "chat.completion", "created": 1, "model": "qwen-plus",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"label":"HARD_NEGATIVE","reason":"different geography"}'}, "finish_reason": "stop"}],
            })
        with openai.OpenAI(api_key="dummy", base_url="https://consumer.invalid/v1", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler), trust_env=False)) as client:
            self.assertEqual(scope["classify_candidate"](client, "qwen-plus", "steel", "steel in CN", "steel in DE"), "HARD_NEGATIVE")
        self.assertEqual(requests[0]["response_format"]["json_schema"]["name"], "hard_negative_label")
        self.assertIn("steel in DE", requests[0]["messages"][1]["content"])


class SupabaseConsumerTest(OfflineTest):
    def test_real_exporter_filters_counts_pagination_errors_and_retry(self):
        consumer = load_module("consumer_export", ROOT / "scripts/tools/export_supabase_json_ordered.py")
        requests = []
        scenario = {"error": False, "transient": False}
        rows = [{"id": str(index), "version": "01.00.000", "json_ordered": {"fixture": index}, "state_code": 100} for index in range(3)]

        def handler(request):
            self.assertEqual(request.url.host, "consumer.invalid")
            requests.append(request)
            if scenario["error"]:
                return httpx.Response(400, request=request, json={"message": "dummy filter rejected", "code": "22000", "details": None, "hint": None})
            if scenario["transient"]:
                scenario["transient"] = False
                return httpx.Response(503, request=request, json={"message": "transient dummy failure"})
            offset, limit = int(request.url.params.get("offset", "0")), int(request.url.params.get("limit", "3"))
            data = [{"id": "0"}] if request.url.params.get("select") == "id" else rows[offset:offset + limit]
            return httpx.Response(200, request=request, json=data, headers={"content-range": f"{offset}-{offset + len(data) - 1}/3"})

        with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as http:
            options = ClientOptions(httpx_client=http, auto_refresh_token=False, persist_session=False)
            with patch.object(consumer, "create_client", side_effect=lambda url, key: create_client(url, key, options=options)):
                client = consumer.get_supabase_client("https://consumer.invalid", "dummy")
                self.assertEqual(consumer.count_rows(client, "flows", [100, 200]), 3)
                self.assertEqual(list(consumer.fetch_rows(client, "flows", [100, 200], 2)), rows)
                with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(consumer.export_table(client, "processes", Path(temporary), [100], 2), 3)
                    saved = sorted((Path(temporary) / "processes").glob("*.json"))
                    self.assertEqual([json.loads(path.read_text()) for path in saved], [{"fixture": index} for index in range(3)])
                    self.assertIsNone(consumer.save_json_ordered({"id": "missing"}, Path(temporary)))
                scenario["error"] = True
                with self.assertRaises(APIError) as error:
                    list(consumer.fetch_rows(client, "flows", [100], 2))
                self.assertEqual(type(error.exception).__name__, "APIError")
                scenario.update(error=False, transient=True)
                with patch("time.sleep", return_value=None):
                    self.assertEqual(list(consumer.fetch_rows(client, "flows", [100], 4)), rows)
                for url, key in [(None, "dummy"), ("https://consumer.invalid", None)]:
                    with self.assertRaises(ValueError):
                        consumer.get_supabase_client(url, key)
        self.assertEqual(requests[0].headers["prefer"], "count=exact")
        self.assertEqual(requests[1].url.params["state_code"], "in.(100,200)")
        self.assertEqual([(request.url.params["offset"], request.url.params["limit"]) for request in requests[1:3]], [("0", "2"), ("2", "2")])
        self.assertEqual(requests[-1].headers["x-retry-count"], "1")
        self.assertTrue(all(request.method == "GET" for request in requests))


class MatplotlibConsumerTest(OfflineTest):
    def test_actual_headless_plots_without_loading_embedding_models(self):
        embedding = types.ModuleType("embedding")
        embedding.get_embedding = lambda *args, **kwargs: self.fail("Model retrieval is outside portable proof")
        with module_override("embedding", embedding), patch.dict(os.environ, {"MPLBACKEND": "Agg"}), \
             patch.object(sys, "path", [str(ROOT / "src"), *sys.path]):
            consumer = load_module("consumer_visualization", ROOT / "src/visualize_similarity.py")
            matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
            with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
                root = Path(temporary)
                consumer.plot_similarity_heatmap(matrix, ["electricity", "water"], str(root / "heatmap.png"))
                with patch.object(consumer, "compute_similarity_matrix", return_value=matrix):
                    result = consumer.plot_multi_model_comparison(["electricity", "water"], ["fixture/first", "fixture/second"], str(root / "multi.png"))
                self.assertEqual(consumer.plt.get_backend().lower(), "agg")
                self.assertEqual(set(result), {"fixture/first", "fixture/second"})
                for actual in result.values():
                    np.testing.assert_array_equal(actual, matrix)
                for name in ["heatmap.png", "multi.png"]:
                    with Image.open(root / name) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertGreater(min(image.size), 100)
                        pixels = np.asarray(image.convert("RGB"))
                        self.assertGreater(float(pixels.std()), 1.0)
                self.assertEqual(consumer.plt.get_fignums(), [])
