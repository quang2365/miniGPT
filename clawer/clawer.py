import requests
import json
import time
import re
from pathlib import Path


# ============================================================
# 1. CONFIG
# ============================================================

API_URL = "https://vi.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "MiniGPTCrawler/1.0"
}


ROOT_CATEGORY = "Trí tuệ nhân tạo"

# Depth:
#
# 0 = chỉ category gốc
# 1 = category gốc + subcategory cấp 1
# 2 = thêm subcategory cấp 2
#
MAX_DEPTH = 2


# Giới hạn số bài để tránh crawl quá lớn khi test.
MAX_ARTICLES = 1000


# Nghỉ giữa các request.
REQUEST_DELAY = 0.3


# File output.
RAW_FILE = Path(
    "data/raw/wikipedia_ai.jsonl"
)

CLEAN_FILE = Path(
    "data/cleaned/wikipedia_ai_clean.jsonl"
)

INPUT_FILE = Path(
    "input.txt"
)


# ============================================================
# 2. SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    HEADERS
)


# ============================================================
# 3. GỌI API
# ============================================================

def api_get(
    params,
    retries=3
):
    """
    Gửi GET request tới MediaWiki API.

    Có retry đơn giản nếu request lỗi.
    """

    last_error = None


    for attempt in range(
        retries
    ):

        try:

            response = session.get(
                API_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            return response.json()


        except Exception as e:

            last_error = e

            print(
                f"Request lỗi "
                f"(attempt {attempt + 1}/{retries}):",
                e
            )

            time.sleep(
                2
            )


    raise last_error


# ============================================================
# 4. LẤY MEMBERS TRONG CATEGORY
# ============================================================

def get_category_members(
    category_name,
    member_type="page"
):
    """
    Lấy toàn bộ members của một Wikipedia category.

    member_type:
        "page"
        "subcat"

    Pagination bằng cmcontinue.
    """

    members = []

    cmcontinue = None


    while True:

        params = {
            "action": "query",

            "list": "categorymembers",

            "cmtitle":
                f"Category:{category_name}",

            # page hoặc subcat
            "cmtype":
                member_type,

            # MediaWiki API hỗ trợ tới 500.
            "cmlimit":
                500,

            "format":
                "json",

            "formatversion":
                2,
        }


        if cmcontinue is not None:

            params["cmcontinue"] = (
                cmcontinue
            )


        data = api_get(
            params
        )


        batch = (
            data
            .get(
                "query",
                {}
            )
            .get(
                "categorymembers",
                []
            )
        )


        members.extend(
            batch
        )


        # Hết pagination.
        if "continue" not in data:
            break


        cmcontinue = (
            data["continue"]
            ["cmcontinue"]
        )


        time.sleep(
            REQUEST_DELAY
        )


    return members


# ============================================================
# 5. RECURSIVE CATEGORY CRAWLER
# ============================================================

def collect_recursive(
    category_name,
    depth,
    max_depth,
    visited_categories,
    visited_pages,
    article_titles,
    max_articles=None
):
    """
    Đi đệ quy:

    category
        ↓
    articles
        ↓
    subcategories
        ↓
    recursive call
    """


    # --------------------------------------------------------
    # STOP: đủ số bài
    # --------------------------------------------------------

    if (
        max_articles is not None
        and
        len(article_titles)
        >= max_articles
    ):
        return


    # --------------------------------------------------------
    # STOP: category đã crawl
    # --------------------------------------------------------

    if (
        category_name
        in
        visited_categories
    ):
        return


    # --------------------------------------------------------
    # STOP: vượt depth
    # --------------------------------------------------------

    if depth > max_depth:
        return


    # Đánh dấu category đã crawl.
    visited_categories.add(
        category_name
    )


    print(
        f"\n[Depth {depth}] "
        f"Category: {category_name}"
    )


    # ========================================================
    # 5.1 LẤY ARTICLES
    # ========================================================

    try:

        pages = get_category_members(
            category_name,
            member_type="page"
        )

    except Exception as e:

        print(
            "Không lấy được pages:",
            category_name,
            e
        )

        pages = []


    for page in pages:

        if (
            max_articles is not None
            and
            len(article_titles)
            >= max_articles
        ):
            return


        title = page.get(
            "title"
        )


        if not title:
            continue


        # Tránh duplicate article.
        if (
            title
            in
            visited_pages
        ):
            continue


        visited_pages.add(
            title
        )


        article_titles.append(
            title
        )


    print(
        "Tổng articles hiện tại:",
        len(article_titles)
    )


    # ========================================================
    # 5.2 DỪNG NẾU ĐẠT MAX DEPTH
    # ========================================================

    if depth == max_depth:
        return


    # ========================================================
    # 5.3 LẤY SUBCATEGORY
    # ========================================================

    try:

        subcategories = (
            get_category_members(
                category_name,
                member_type="subcat"
            )
        )

    except Exception as e:

        print(
            "Không lấy được subcategory:",
            category_name,
            e
        )

        return


    print(
        "Subcategories:",
        len(subcategories)
    )


    # ========================================================
    # 5.4 RECURSION
    # ========================================================

    for subcategory in subcategories:

        if (
            max_articles is not None
            and
            len(article_titles)
            >= max_articles
        ):
            return


        title = subcategory.get(
            "title",
            ""
        )


        # API thường trả:
        # Category:Học máy
        #
        # Ta cần:
        # Học máy

        prefix = "Category:"


        if title.startswith(
            prefix
        ):

            sub_name = (
                title[
                    len(prefix):
                ]
            )

        else:

            sub_name = title


        if not sub_name:
            continue


        collect_recursive(
            category_name=
                sub_name,

            depth=
                depth + 1,

            max_depth=
                max_depth,

            visited_categories=
                visited_categories,

            visited_pages=
                visited_pages,

            article_titles=
                article_titles,

            max_articles=
                max_articles
        )


# ============================================================
# 6. ENTRY POINT ĐỂ LẤY ARTICLE TITLES
# ============================================================

def collect_article_titles(
    root_category,
    max_depth=2,
    max_articles=None
):
    """
    Root category
        ↓
    recursive crawl
        ↓
    unique article titles
    """

    visited_categories = set()

    visited_pages = set()

    article_titles = []


    collect_recursive(
        category_name=
            root_category,

        depth=
            0,

        max_depth=
            max_depth,

        visited_categories=
            visited_categories,

        visited_pages=
            visited_pages,

        article_titles=
            article_titles,

        max_articles=
            max_articles
    )


    print(
        "\n================================"
    )

    print(
        "Tổng category đã crawl:",
        len(
            visited_categories
        )
    )

    print(
        "Tổng article unique:",
        len(
            article_titles
        )
    )

    print(
        "================================"
    )


    return article_titles


# ============================================================
# 7. LẤY NỘI DUNG MỘT ARTICLE
# ============================================================

def get_article_text(
    title
):
    """
    Lấy plain text của một bài Wikipedia.

    Output:

    {
        "pageid": ...,
        "title": ...,
        "text": ...
    }
    """

    params = {
        "action":
            "query",

        "prop":
            "extracts",

        # Plain text.
        "explaintext":
            1,

        # Đưa section heading về dạng dễ đọc hơn.
        "exsectionformat":
            "plain",

        "titles":
            title,

        "format":
            "json",

        "formatversion":
            2,
    }


    data = api_get(
        params
    )


    pages = (
        data
        .get(
            "query",
            {}
        )
        .get(
            "pages",
            []
        )
    )


    if not pages:
        return None


    page = pages[0]


    if (
        page.get(
            "missing"
        )
        is True
    ):
        return None


    text = page.get(
        "extract",
        ""
    )


    return {
        "pageid":
            page.get(
                "pageid"
            ),

        "title":
            page.get(
                "title",
                title
            ),

        "text":
            text
    }


# ============================================================
# 8. CRAWL TOÀN BỘ ARTICLE → RAW JSONL
# ============================================================

def crawl_articles(
    titles,
    output_file
):
    """
    Article titles
        ↓
    get_article_text()
        ↓
    raw JSONL
    """


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    saved = 0


    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:


        for i, title in enumerate(
            titles,
            start=1
        ):


            try:

                article = (
                    get_article_text(
                        title
                    )
                )


                if article is None:
                    continue


                text = (
                    article
                    .get(
                        "text",
                        ""
                    )
                    .strip()
                )


                # Bỏ bài gần như không có nội dung.
                if len(text) < 200:
                    continue


                f.write(
                    json.dumps(
                        article,
                        ensure_ascii=False
                    )
                    + "\n"
                )


                saved += 1


                print(
                    f"[{i}/{len(titles)}] "
                    f"{title} | "
                    f"{len(text):,} chars"
                )


            except Exception as e:

                print(
                    "ERROR ARTICLE:",
                    title,
                    e
                )


            time.sleep(
                REQUEST_DELAY
            )


    print(
        "\nRaw articles saved:",
        saved
    )


# ============================================================
# 9. CLEAN TEXT
# ============================================================

def clean_text(
    text
):
    """
    Cleaning nhẹ cho LLM corpus.

    Không làm sạch quá mạnh để tránh mất
    cấu trúc ngôn ngữ.
    """


    if not text:
        return ""


    # --------------------------------------------------------
    # Unicode whitespace
    # --------------------------------------------------------

    text = text.replace(
        "\u00a0",
        " "
    )


    # --------------------------------------------------------
    # Normalize newline
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    # --------------------------------------------------------
    # Clean từng dòng
    # --------------------------------------------------------

    clean_lines = []


    for line in text.split(
        "\n"
    ):

        line = line.strip()


        # Space/tab liên tiếp → một space.
        line = re.sub(
            r"[ \t]+",
            " ",
            line
        )


        # Bỏ dòng trống.
        if not line:
            continue


        # Bỏ dòng cực ngắn.
        if len(line) < 3:
            continue


        clean_lines.append(
            line
        )


    text = "\n".join(
        clean_lines
    )


    # --------------------------------------------------------
    # Quá nhiều newline → 2 newline
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# 10. NORMALIZE CHO DEDUP
# ============================================================

def normalize_for_dedup(
    text
):
    """
    Normalize nhẹ để phát hiện exact-ish duplicates.

    Không dùng output này để training.
    Chỉ dùng làm key dedup.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 11. CLEAN + DEDUP
# ============================================================

def clean_and_deduplicate(
    raw_file,
    clean_file
):
    """
    raw JSONL
        ↓
    clean
        ↓
    dedup
        ↓
    clean JSONL
    """


    clean_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    seen_pageids = set()

    seen_texts = set()


    total = 0

    kept = 0

    duplicates = 0


    with raw_file.open(
        "r",
        encoding="utf-8"
    ) as src, clean_file.open(
        "w",
        encoding="utf-8"
    ) as dst:


        for line in src:

            if not line.strip():
                continue


            total += 1


            article = json.loads(
                line
            )


            pageid = article.get(
                "pageid"
            )


            title = article.get(
                "title",
                ""
            ).strip()


            text = clean_text(
                article.get(
                    "text",
                    ""
                )
            )


            # Bỏ text quá ngắn.
            if len(text) < 200:
                continue


            # -----------------------------------------------
            # Dedup bằng pageid.
            # -----------------------------------------------

            if (
                pageid is not None
                and
                pageid in seen_pageids
            ):

                duplicates += 1

                continue


            if pageid is not None:

                seen_pageids.add(
                    pageid
                )


            # -----------------------------------------------
            # Dedup bằng normalized text.
            # -----------------------------------------------

            text_key = (
                normalize_for_dedup(
                    text
                )
            )


            if (
                text_key
                in
                seen_texts
            ):

                duplicates += 1

                continue


            seen_texts.add(
                text_key
            )


            cleaned_article = {
                "pageid":
                    pageid,

                "title":
                    title,

                "text":
                    text
            }


            dst.write(
                json.dumps(
                    cleaned_article,
                    ensure_ascii=False
                )
                + "\n"
            )


            kept += 1


    print(
        "\n===== CLEAN RESULT ====="
    )

    print(
        "Raw articles:",
        total
    )

    print(
        "Kept:",
        kept
    )

    print(
        "Duplicates:",
        duplicates
    )


# ============================================================
# 12. BUILD INPUT.TXT
# ============================================================

def build_input_txt(
    clean_file,
    output_file
):
    """
    clean JSONL
        ↓
    concatenate articles
        ↓
    input.txt
    """


    article_count = 0

    total_chars = 0


    with clean_file.open(
        "r",
        encoding="utf-8"
    ) as src, output_file.open(
        "w",
        encoding="utf-8"
    ) as dst:


        for line in src:

            if not line.strip():
                continue


            article = json.loads(
                line
            )


            title = (
                article
                .get(
                    "title",
                    ""
                )
                .strip()
            )


            text = (
                article
                .get(
                    "text",
                    ""
                )
                .strip()
            )


            if not text:
                continue


            # -----------------------------------------------
            # Article separator
            # -----------------------------------------------

            dst.write(
                "\n"
            )


            # -----------------------------------------------
            # Title
            # -----------------------------------------------

            if title:

                dst.write(
                    title
                )

                dst.write(
                    "\n"
                )


            # -----------------------------------------------
            # Content
            # -----------------------------------------------

            dst.write(
                text
            )


            # Hai newline giữa các article.
            dst.write(
                "\n\n"
            )


            article_count += 1


            total_chars += (
                len(title)
                +
                len(text)
                +
                3
            )


    print(
        "\n===== INPUT.TXT ====="
    )

    print(
        "Articles:",
        article_count
    )

    print(
        "Characters:",
        f"{total_chars:,}"
    )

    print(
        "Output:",
        output_file
    )


# ============================================================
# 13. SANITY CHECK INPUT
# ============================================================

def sanity_check(
    input_file
):
    """
    In vài statistics để kiểm tra corpus.
    """

    text = input_file.read_text(
        encoding="utf-8"
    )


    print(
        "\n===== SANITY CHECK ====="
    )


    print(
        "Characters:",
        f"{len(text):,}"
    )


    print(
        "Lines:",
        f"{len(text.splitlines()):,}"
    )


    print(
        "File size:",
        f"{input_file.stat().st_size / 1024 / 1024:.2f} MB"
    )


    print(
        "\n===== SAMPLE =====\n"
    )


    print(
        text[:3000]
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "======================================="
    )

    print(
        "STEP 1 — COLLECT ARTICLE TITLES"
    )

    print(
        "======================================="
    )


    titles = collect_article_titles(
        root_category=
            ROOT_CATEGORY,

        max_depth=
            MAX_DEPTH,

        max_articles=
            MAX_ARTICLES
    )


    # --------------------------------------------------------
    # In thử title
    # --------------------------------------------------------

    print(
        "\n20 bài đầu:"
    )


    for title in titles[:20]:

        print(
            "-",
            title
        )


    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n======================================="
    )

    print(
        "STEP 2 — DOWNLOAD ARTICLES"
    )

    print(
        "======================================="
    )


    crawl_articles(
        titles=
            titles,

        output_file=
            RAW_FILE
    )


    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n======================================="
    )

    print(
        "STEP 3 — CLEAN + DEDUP"
    )

    print(
        "======================================="
    )


    clean_and_deduplicate(
        raw_file=
            RAW_FILE,

        clean_file=
            CLEAN_FILE
    )


    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n======================================="
    )

    print(
        "STEP 4 — BUILD INPUT.TXT"
    )

    print(
        "======================================="
    )


    build_input_txt(
        clean_file=
            CLEAN_FILE,

        output_file=
            INPUT_FILE
    )


    # ========================================================
    # STEP 5
    # ========================================================

    print(
        "\n======================================="
    )

    print(
        "STEP 5 — SANITY CHECK"
    )

    print(
        "======================================="
    )


    sanity_check(
        INPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()