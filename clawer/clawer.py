import argparse
import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse, quote
from xml.etree import ElementTree as ET

import requests

API_URL = "https://vi.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MiniGPTCrawler/2.0"}
ROOT_CATEGORY = "Trí tuệ nhân tạo"
MAX_DEPTH = 2
MAX_ARTICLES = 1000
REQUEST_DELAY = 0.3
# Wikipedia dump is an official, versioned Wikimedia export mirrored on HF;
# unlike the MediaWiki API it is reproducible and avoids API rate limits.
DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_DATASET_CONFIG = "20231101.vi"
TRUSTED_DATASETS = {
    "wikimedia/wikipedia": "Wikimedia Wikipedia dump",
    "allenai/c4": "AllenAI C4 corpus",
    "HuggingFaceFW/fineweb-edu": "Hugging Face FineWeb-Edu",
}
RAW_FILE = Path("data/raw/vietnamese_corpus.jsonl")
CLEAN_FILE = Path("data/cleaned/vietnamese_corpus_clean.jsonl")
INPUT_FILE = Path("input.txt")
MAX_BYTES = 20 * 1024 * 1024
session = requests.Session()
session.headers.update(HEADERS)


def fetch(url, params=None, retries=3):
    """Giới hạn dung lượng cả khi máy chủ không gửi Content-Length."""
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("URL phải dùng http hoặc https")
    for attempt in range(retries):
        try:
            time.sleep(REQUEST_DELAY)
            with session.get(url, params=params, timeout=(10, 60), stream=True) as r:
                r.raise_for_status()
                chunks, size = [], 0
                for chunk in r.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"Nguồn vượt giới hạn {MAX_BYTES} bytes")
                    chunks.append(chunk)
                return b"".join(chunks), r.headers.get("Content-Type", ""), r.url
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise ValueError("retries phải lớn hơn 0")


def api_get(params, retries=3):
    body, _, _ = fetch(API_URL, params, retries)
    data = json.loads(body)
    if "error" in data:
        raise ValueError(f"MediaWiki: {data['error']}")
    return data


def get_category_members(category_name, member_type="page"):
    params = dict(action="query", list="categorymembers",
                  cmtitle=category_name if ":" in category_name else f"Category:{category_name}",
                  cmtype=member_type, cmlimit=500, format="json", formatversion=2)
    if member_type == "page":
        params["cmnamespace"] = 0
    members = []
    while True:
        data = api_get(params)
        members.extend(data.get("query", {}).get("categorymembers", []))
        if "continue" not in data:
            return members
        params.update(data["continue"])


def collect_recursive(category_name, depth, max_depth, visited_categories,
                      visited_pages, article_titles, max_articles=None):
    if depth > max_depth or category_name in visited_categories:
        return
    if max_articles is not None and len(article_titles) >= max_articles:
        return
    visited_categories.add(category_name)
    for page in get_category_members(category_name):
        title = page.get("title")
        if title and title not in visited_pages:
            visited_pages.add(title)
            article_titles.append(title)
        if max_articles is not None and len(article_titles) >= max_articles:
            return
    if depth < max_depth:
        for child in get_category_members(category_name, "subcat"):
            collect_recursive(child["title"], depth + 1, max_depth,
                              visited_categories, visited_pages, article_titles, max_articles)


def collect_article_titles(root_category, max_depth=2, max_articles=None):
    titles = []
    collect_recursive(root_category, 0, max_depth, set(), set(), titles, max_articles)
    return titles


def record(title, text, source, kind, **metadata):
    return dict(title=str(title), text=text, source=source, source_type=kind,
                fetched_at=datetime.now(timezone.utc).isoformat(), **metadata)


def get_article_text(title):
    data = api_get(dict(action="query", prop="extracts", explaintext=1,
                        exsectionformat="plain", titles=title, redirects=1,
                        format="json", formatversion=2))
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None
    page = pages[0]
    return record(page.get("title", title), page.get("extract", ""), API_URL,
                  "wikipedia", pageid=page.get("pageid"),
                  url=API_URL.split('/w/api.php')[0] + '/wiki/' + quote(page.get('title', title)))


def html_text(data):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Đọc HTML cần cài beautifulsoup4") from exc
    soup = BeautifulSoup(data, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for node in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        node.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    return title, body.get_text("\n", strip=True)


def decode_text(data, content_type=""):
    match = re.search(r'charset=["\']?([^;\s"\']+)', content_type, re.I)
    encoding = match.group(1) if match else "utf-8-sig"
    return data.decode(encoding)  # Báo lỗi encoding thay vì âm thầm làm hỏng tiếng Việt.


def structured_text(value):
    """Giữ cặp key/value, danh sách và dữ liệu hội thoại, không chỉ một field text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_document(data, source, content_type="", format_hint=None):
    ext = (format_hint or Path(urlparse(source).path).suffix.lstrip('.')).lower()
    mime = content_type.split(';')[0].strip().lower()
    kinds = {"application/pdf": "pdf", "text/html": "html", "application/json": "json",
             "text/csv": "csv", "application/rss+xml": "xml", "application/atom+xml": "xml",
             "application/xml": "xml", "text/xml": "xml", "text/plain": "txt",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx"}
    if not format_hint and mime in kinds:
        ext = kinds[mime]
    title = Path(urlparse(source).path).name or source
    if ext in {"html", "htm"}:
        heading, text = html_text(data)
        yield record(heading or title, text, source, "html")
    elif ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Đọc PDF cần cài pypdf") from exc
        reader = PdfReader(io.BytesIO(data))
        text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
        if not text.strip():
            raise ValueError("PDF không có lớp văn bản; chưa hỗ trợ OCR")
        yield record(title, text, source, ext)
    elif ext == "docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_BYTES:
                raise ValueError("DOCX giải nén vượt giới hạn")
            root = ET.fromstring(archive.read(info))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = '\n'.join(''.join(p.itertext()) for p in root.findall('.//w:p', ns))
        yield record(title, text, source, ext)
    elif ext in {"xml", "rss", "atom"}:
        root = ET.fromstring(data)
        entries = [node for node in root.iter() if node.tag.split('}')[-1] in {"item", "entry"}]
        if not entries:
            yield record(title, '\n'.join(t.strip() for t in root.itertext() if t.strip()), source, "xml")
        for index, entry in enumerate(entries):
            fields = {n.tag.split('}')[-1]: ''.join(n.itertext()) for n in entry}
            body = fields.get('encoded') or fields.get('content') or fields.get('description') or fields.get('summary', '')
            if '<' in body and '>' in body:
                _, body = html_text(body)
            yield record(fields.get('title', title), body, source, "feed", entry_index=index)
    elif ext in {"json", "jsonl", "ndjson"}:
        text = decode_text(data, content_type)
        rows = (json.loads(line) for line in text.splitlines() if line.strip()) if ext != "json" else json.loads(text)
        if ext == "json" and not isinstance(rows, list):
            rows = [rows]
        for index, row in enumerate(rows):
            heading = row.get('title', title) if isinstance(row, dict) else title
            yield record(heading, structured_text(row), source, ext, row_index=index)
    elif ext in {"csv", "tsv"}:
        rows = csv.DictReader(io.StringIO(decode_text(data, content_type)), delimiter='\t' if ext == 'tsv' else ',')
        for index, row in enumerate(rows):
            yield record(title, structured_text(row), source, ext, row_index=index)
    elif ext in {"txt", "md", "markdown", "rst"}:
        yield record(title, decode_text(data, content_type), source, ext)
    else:
        raise ValueError(f"Định dạng chưa hỗ trợ: {ext or mime}; dùng --format nếu URL không có đuôi")


def crawl_articles(titles, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as dst:
        for title in titles:
            try:
                article = get_article_text(title)
                if article and article['text'].strip():
                    dst.write(json.dumps(article, ensure_ascii=False) + '\n')
            except Exception as exc:
                print(f"Lỗi bài {title}: {exc}")


def clean_text(text):
    text = unicodedata.normalize('NFC', text).replace('\u00a0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    boilerplate = {
        'tham khảo', 'liên kết ngoài', 'xem thêm', 'chú thích',
        'tài liệu tham khảo', 'đọc thêm', 'thư mục', 'nguồn',
    }
    raw_lines = [re.sub(r'[ \t]+', ' ', raw).strip()
                 for raw in text.split('\n')]
    local_counts = Counter(
        normalize_for_dedup(line)
        for line in raw_lines
        if line
    )
    lines = []
    for line in raw_lines:
        if not line:
            if lines and lines[-1] != '':
                lines.append('')
            continue
        normalized = normalize_for_dedup(line)
        if normalized in boilerplate:
            continue
        # Wikipedia extracts often contain a standalone link marker or a
        # one-word navigation label repeated many times.
        if re.fullmatch(r'(?:liên kết ngoài|tham khảo|xem thêm)(?:\s*\[\d+\])?', normalized):
            continue
        if (len(line) < 24 and local_counts[normalized] >= 2 and
                not re.search(r'[.!?:;,)]', line)):
            continue
        lines.append(line)
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def normalize_for_dedup(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def clean_and_deduplicate(raw_file, clean_file, min_chars=200):
    clean_file = Path(clean_file)
    clean_file.parent.mkdir(parents=True, exist_ok=True)
    seen, kept = set(), 0
    with Path(raw_file).open(encoding='utf-8') as src, clean_file.open('w', encoding='utf-8') as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            text = clean_text(item.get('text', ''))
            key = hashlib.sha256(normalize_for_dedup(text).encode('utf-8')).hexdigest()
            if len(text) < min_chars or not text or key in seen:
                continue
            seen.add(key)
            item.update(text=text, content_hash=key)
            dst.write(json.dumps(item, ensure_ascii=False) + '\n')
            kept += 1
    print(f"Sau clean/dedup: {kept} bản ghi")


def build_input_txt(clean_file, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Path(clean_file).open(encoding='utf-8') as src, output_file.open('w', encoding='utf-8') as dst:
        for line in src:
            if line.strip():
                item = json.loads(line)
                dst.write(f"{item.get('title', '')}\n{item['text']}\n\n")


def sanity_check(input_file):
    print(f"Output: {input_file} | {Path(input_file).stat().st_size:,} bytes")


def dataset_documents(dataset_id, *, config=None, split="train", domains=(),
                      max_scan=100000, limit=1000):
    """Đọc từng bản ghi; không tải toàn bộ corpus vào RAM/ổ đĩa.

max_scan giới hạn số dòng xét khi lọc domain; limit giới hạn dòng trả về.
Giới hạn MAX_BYTES của fetch chỉ áp dụng URL/tệp, không áp dụng stream HF.
"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Nguồn Hugging Face cần datasets: python -m pip install datasets"
        ) from exc
    if limit <= 0:
        return
    options = dict(split=split, streaming=True)
    if config:
        options["name"] = config
    rows = load_dataset(dataset_id, **options)
    wanted = {value.casefold() for value in domains}
    saved = 0
    for index, row in enumerate(islice(rows, max_scan)):
        if "text" not in row or (wanted and "domain" not in row):
            raise ValueError("Dataset phải có cột text và cột domain nếu dùng --domain")
        domain = row.get("domain")
        if wanted and str(domain).casefold() not in wanted:
            continue
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError(f"Cột text ở dòng {index} không phải chuỗi")
        if not text.strip():
            continue
        yield record(row.get("title") or "", text,
                     f"https://huggingface.co/datasets/{dataset_id}", "huggingface",
                     dataset_id=dataset_id, dataset_config=config, split=split,
                     row_index=index, original_id=row.get("id"), domain=domain)
        saved += 1
        if saved >= limit:
            return
    print(f"Dataset {dataset_id}: kết thúc luồng hoặc đạt --max-scan {max_scan}; "
          f"lấy được {saved} bản ghi phù hợp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', action='append', help='Lặp lại để thêm chủ đề Wikipedia')
    parser.add_argument('--dataset', help='Dataset Hugging Face; mặc định Wikimedia Wikipedia dump tiếng Việt')
    parser.add_argument('--dataset-config', help='Tên config nếu dataset yêu cầu')
    parser.add_argument('--split', default='train')
    parser.add_argument('--domain', action='append', default=[], help='Lọc nhãn lĩnh vực HF, ví dụ Science; lặp lại để chọn nhiều')
    parser.add_argument('--max-scan', type=int, default=100000, help='Số dòng HF tối đa xét trước khi lọc lĩnh vực')
    parser.add_argument('--url', action='append', default=[], help='URL web/tệp/feed trực tiếp')
    parser.add_argument('--file', action='append', default=[], help='Đường dẫn tệp cục bộ')
    parser.add_argument('--format', help='Định dạng ép buộc cho các URL/tệp')
    parser.add_argument('--max-depth', type=int, default=MAX_DEPTH)
    parser.add_argument('--max-articles', type=int, default=MAX_ARTICLES, help='Giới hạn tổng bản ghi raw')
    parser.add_argument('--min-chars', type=int, default=200)
    parser.add_argument('--raw-file', type=Path, default=RAW_FILE)
    parser.add_argument('--clean-file', type=Path, default=CLEAN_FILE)
    parser.add_argument('--output', type=Path, default=INPUT_FILE)
    parser.add_argument('--overwrite', action='store_true', help='Cho phép ghi đè output đã tồn tại')
    parser.add_argument('--trusted-only', action='store_true',
                        help='Chỉ cho phép dataset trong danh sách nguồn đã kiểm duyệt')
    args = parser.parse_args()
    if args.max_depth < 0 or args.max_articles < 1 or args.min_chars < 0 or args.max_scan < 1:
        parser.error('Giới hạn không hợp lệ')
    dataset_id = args.dataset
    if not (dataset_id or args.category or args.url or args.file):
        dataset_id = DEFAULT_DATASET
        if not args.dataset_config:
            args.dataset_config = DEFAULT_DATASET_CONFIG
    if args.trusted_only and dataset_id not in TRUSTED_DATASETS:
        parser.error('--trusted-only yêu cầu dataset nằm trong danh sách nguồn tin cậy')
    if (args.domain or args.dataset_config) and not dataset_id:
        parser.error('--domain/--dataset-config cần nguồn --dataset; --category vẫn gọi Wikipedia')
    if dataset_id:
        # Kiểm tra dependency trước khi mở output, kể cả khi cho phép overwrite.
        try:
            import datasets  # noqa: F401
        except ImportError:
            parser.error('Thiếu datasets. Cài bằng: python -m pip install datasets')
    outputs = [args.raw_file, args.clean_file, args.output]
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != 3 or set(resolved) & {Path(p).resolve() for p in args.file}:
        parser.error('Output phải khác nhau và không trùng tệp nguồn')
    if not args.overwrite and any(p.exists() for p in outputs):
        parser.error('Output đã tồn tại; chọn đường dẫn khác hoặc dùng --overwrite')
    categories = args.category or []
    count, errors = 0, 0

    def sources():
        if dataset_id:
            yield 'dataset', dataset_id
        for category in categories:
            # Giới hạn riêng việc khám phá mỗi category.
            yield 'category', category
        for url in args.url:
            yield 'url', url
        for path in args.file:
            yield 'file', path

    def documents(kind, source):
        if kind == 'dataset':
            yield from dataset_documents(source, config=args.dataset_config,
                                         split=args.split, domains=args.domain,
                                         max_scan=args.max_scan,
                                         limit=args.max_articles - count)
        elif kind == 'category':
            for title in collect_article_titles(source, args.max_depth, args.max_articles - count):
                try:
                    item = get_article_text(title)
                    if item:
                        yield item
                except Exception as exc:
                    print(f'Lỗi bài {title}: {exc}')
        elif kind == 'url':
            data, mime, final_url = fetch(source)
            yield from parse_document(data, final_url, mime, args.format)
        else:
            path = Path(source).resolve()
            with path.open('rb') as handle:
                data = handle.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError('Tệp vượt giới hạn dung lượng')
            yield from parse_document(data, path.as_uri(), format_hint=args.format)

    args.raw_file.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_file.open('w', encoding='utf-8') as dst:
        for kind, source in sources():
            if count >= args.max_articles:
                break
            try:
                for item in documents(kind, source):
                    if not item['text'].strip():
                        continue
                    dst.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 100 == 0:
                        print(f'Đã lưu {count:,}/{args.max_articles:,} bản ghi raw')
                    if count >= args.max_articles:
                        break
            except Exception as exc:
                errors += 1
                print(f'Lỗi nguồn {source}: {exc}')
    if not count:
        raise SystemExit('Không có bản ghi; không tạo/ghi đè cleaned và input.txt')
    clean_and_deduplicate(args.raw_file, args.clean_file, args.min_chars)
    build_input_txt(args.clean_file, args.output)
    print(f'Đã lưu {count} bản ghi raw; {errors} nguồn lỗi')
    sanity_check(args.output)


if __name__ == '__main__':
    main()
