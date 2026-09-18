"""Chuẩn hóa bộ prompt/response tiếng Việt thành user/assistant JSONL."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


SYSTEM_MESSAGE = "Bạn là trợ lý AI. Hãy trả lời rõ ràng, hữu ích và lịch sự bằng tiếng Việt."
USER_PREFIX = "Yêu cầu của người dùng: "
END_CHARS = ".!?。！？…:;)]}" 


def clean(value: object) -> str:
    """Chuẩn hóa whitespace nhưng không sửa nội dung hay dấu câu gốc."""
    return re.sub(r"[ \t]+", " ", str(value or "")).strip()


def enrich_user_prompt(prompt: str) -> str:
    prompt = clean(prompt)
    if prompt.startswith(USER_PREFIX):
        return prompt
    return USER_PREFIX + prompt


def polish_assistant_response(response: str) -> str:
    response = clean(response)
    if not response:
        return "Mình chưa có đủ thông tin để trả lời chính xác. Vui lòng cung cấp thêm chi tiết."
    # Response nguồn thường bị cắt ở giữa câu. Đánh dấu bằng ellipsis thay vì
    # tự bịa phần kiến thức còn thiếu.
    if response[-1] not in END_CHARS:
        response += "…"
    return response


def transform(input_path: Path, output_path: Path, backup: bool = True) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input và output phải khác nhau để tránh mất dữ liệu khi lỗi")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if backup and output_path.exists():
        backup_path = output_path.with_suffix(output_path.suffix + ".bak")
        backup_path.write_bytes(output_path.read_bytes())

    total = 0
    changed = 0
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.stem + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with input_path.open("r", encoding="utf-8") as src, temporary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as dst:
            for line_number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or "prompt" not in item or "response" not in item:
                    raise ValueError(f"Dòng {line_number} thiếu prompt/response")
                user = enrich_user_prompt(item["prompt"])
                assistant = polish_assistant_response(item["response"])
                dst.write(json.dumps({
                    "system": SYSTEM_MESSAGE,
                    "user": user,
                    "assistant": assistant,
                }, ensure_ascii=False) + "\n")
                total += 1
                changed += int(user != item["prompt"] or assistant != item["response"])
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    total, changed = transform(args.input, args.output, backup=not args.no_backup)
    print(f"Processed {total:,} lines; normalized {changed:,} lines.")


if __name__ == "__main__":
    main()
