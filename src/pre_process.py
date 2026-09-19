"""将 TIDAS JSON 文件转换为 Markdown 格式"""

from pathlib import Path
from typing import Optional

from tidas_sdk import create_process_from_json

from src.utils import tidas_process_to_markdown

ROOT_PATH = Path(__file__).parent.parent
DEFAULT_INPUT_DIR = ROOT_PATH / "data" / "tidas" / "processes"
DEFAULT_OUTPUT_DIR = ROOT_PATH / "data" / "tidas" / "markdown"


def convert_single_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    lang: str = "en",
) -> Path:
    """将单个 TIDAS JSON 文件转换为 Markdown

    Args:
        input_path: JSON 文件路径
        output_path: 输出 Markdown 文件路径，如果不指定则自动生成
        lang: 首选语言，默认英文

    Returns:
        输出文件路径
    """
    process = create_process_from_json(input_path)
    markdown = tidas_process_to_markdown(process, lang)
    print(markdown)

    if output_path is None:
        output_path = input_path.with_suffix(".md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return output_path


def convert_all_files(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    lang: str = "en",
) -> dict:
    """批量将目录下所有 TIDAS JSON 文件转换为 Markdown

    Args:
        input_dir: 输入目录，默认为 data/tidas/processes
        output_dir: 输出目录，默认为 data/tidas/markdown
        lang: 首选语言，默认英文

    Returns:
        包含转换结果的字典：
        {
            "success": [(input_path, output_path), ...],
            "failed": [(input_path, error_msg), ...]
        }
    """
    if input_dir is None:
        input_dir = DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    results = {"success": [], "failed": []}

    print(f"找到 {len(json_files)} 个 JSON 文件")

    for json_file in json_files:
        output_path = output_dir / f"{json_file.stem}.md"
        try:
            convert_single_file(json_file, output_path, lang)
            results["success"].append((json_file, output_path))
            print(f"✓ {json_file.name} -> {output_path.name}")
        except Exception as e:
            results["failed"].append((json_file, str(e)))
            print(f"✗ {json_file.name}: {e}")

    print(f"\n转换完成: 成功 {len(results['success'])}, 失败 {len(results['failed'])}")
    return results


if __name__ == "__main__":
    file_path = ROOT_PATH / "data" / "tidas" / "processes" / "0a0bf34e-6af3-3f63-a1f5-6032e1eca2e5.json"
    output_path = ROOT_PATH / "data" / "markdown" / "processes" / f"{file_path.stem}.md"
    convert_single_file(file_path, output_path=output_path)
