"""通用 Markdown 生成脚本：支持 flows/processes 的 JSON（含 Supabase 导出或本地 TIDAS）。"""

import json
import signal
import sys
from pathlib import Path
from typing import Iterable, Optional

from tidas_sdk import create_flow_from_json, create_process_from_json
from tidas_sdk.entities.utils import default_timestamp
from tqdm import tqdm

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_PATH))

DEFAULT_INPUT_DIR = ROOT_PATH / "data" / "tidas" / "processes"
DEFAULT_OUTPUT_DIR = ROOT_PATH / "data" / "markdown"


class ProgressTracker:
    """Track progress and handle interruption gracefully."""

    def __init__(self, total: int, start_index: int = 0):
        self.total = total
        self.start_index = start_index
        self.current_index = start_index
        self.success_count = 0
        self.failed_count = 0
        self._interrupted = False

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal and print resume information."""
        self._interrupted = True
        print("\n" + "=" * 60)
        print("INTERRUPTED!")
        print("=" * 60)
        print(f"Progress: {self.current_index}/{self.total}")
        print(f"Successfully processed: {self.success_count}")
        print(f"Failed: {self.failed_count}")
        print("-" * 60)
        print(f"To resume, run with --start-index {self.current_index}")
        print("=" * 60)
        sys.exit(1)

    def update(self, success: bool):
        """Update progress after processing a file."""
        self.current_index += 1
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1


def _ensure_admin_timestamp(container: dict, admin_key: str = "administrativeInformation") -> None:
    """Ensure administrativeInformation.dataEntryBy.common:timeStamp exists and is a mapping."""
    admin = container.get(admin_key)
    if not isinstance(admin, dict):
        admin = {}
        container[admin_key] = admin
    data_entry = admin.get("dataEntryBy")
    if not isinstance(data_entry, dict):
        data_entry = {}
        admin["dataEntryBy"] = data_entry
    if "common:timeStamp" not in data_entry or not data_entry.get("common:timeStamp"):
        data_entry["common:timeStamp"] = default_timestamp().isoformat()


def _sanitize_payload(payload: dict) -> tuple[str, str]:
    """Return entity type and sanitized JSON string."""
    if "flowDataSet" in payload:
        flow_ds = payload.get("flowDataSet")
        if isinstance(flow_ds, dict):
            _ensure_admin_timestamp(flow_ds)
        return "flow", json.dumps(payload)
    if "processDataSet" in payload:
        proc_ds = payload.get("processDataSet")
        if isinstance(proc_ds, dict):
            _ensure_admin_timestamp(proc_ds)
        return "process", json.dumps(payload)
    raise ValueError("无法识别的数据集类型")


def convert_json_to_markdown(json_path: Path, output_dir: Path, lang: str = "en") -> Path:
    """Convert a single JSON (flow/process) to Markdown. Output goes to flows/ or processes/."""
    raw = json_path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    entity: str
    sanitized_json: str
    entity, sanitized_json = _sanitize_payload(payload)

    if entity == "flow":
        obj = create_flow_from_json(sanitized_json)
    else:
        obj = create_process_from_json(sanitized_json)

    markdown = obj.to_markdown(lang=lang)

    target_dir = output_dir / (f"{entity}s")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{json_path.stem}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    """Yield JSON files. If flows/ or processes/ subdirs exist, traverse them; else scan root."""
    flows_dir = input_dir / "flows"
    procs_dir = input_dir / "processes"
    if flows_dir.exists() or procs_dir.exists():
        if flows_dir.exists():
            yield from sorted(flows_dir.glob("*.json"))
        if procs_dir.exists():
            yield from sorted(procs_dir.glob("*.json"))
    else:
        yield from sorted(input_dir.glob("*.json"))


def generate_markdown_files(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    lang: str = "en",
    start_index: int = 0,
) -> dict:
    """Generate markdown files from JSON (flows/processes)."""
    if input_dir is None:
        input_dir = DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    json_files = list(iter_json_files(input_dir))
    total_files = len(json_files)

    print(f"Found {total_files} JSON files")
    print(f"Output directory: {output_dir}")

    if start_index > 0:
        print(f"Resuming from index {start_index}")
        if start_index >= total_files:
            print(f"Error: start_index ({start_index}) >= total files ({total_files})")
            return {"success": 0, "failed": 0}

    tracker = ProgressTracker(total_files, start_index)

    files_to_process = json_files[start_index:]
    pbar = tqdm(
        enumerate(files_to_process, start=start_index),
        total=len(files_to_process),
        desc="Generating markdown",
    )

    for idx, json_file in pbar:
        tracker.current_index = idx
        pbar.set_postfix(
            {
                "file": json_file.stem[:20],
                "ok": tracker.success_count,
                "fail": tracker.failed_count,
            }
        )

        try:
            convert_json_to_markdown(json_file, output_dir, lang)
            tracker.update(success=True)
        except Exception as e:
            tracker.update(success=False)
            tqdm.write(f"✗ [{idx}] {json_file}: {e}")

    print("\n" + "=" * 60)
    print("Markdown generation complete")
    print("=" * 60)
    print(f"  Success: {tracker.success_count}")
    print(f"  Failed: {tracker.failed_count}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    return {"success": tracker.success_count, "failed": tracker.failed_count}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate markdown files from flow/process JSON (Supabase export or TIDAS)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing JSON files or flows/processes subdirs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for markdown files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Preferred language for markdown generation (default: en)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index to start processing from, for resuming (default: 0)",
    )

    args = parser.parse_args()

    generate_markdown_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    """
    uv run python scripts/step1_generate_markdown.py \
  --input-dir data/supabase_exports \
  --output-dir data/supabase_exports_markdown \
  --lang en
    """
    main()
