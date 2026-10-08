# src/tools/web.py
import re

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from markdownify import markdownify as to_markdown
import pymupdf
import pymupdf4llm

import dotenv

from src.tools.base import make_tool as _tool

EMAIL = dotenv.dotenv_values().get("EMAIL", "")
HEADERS = {
    "User-Agent": f"gaia-agent/0.1 (educational project; contact: {EMAIL})"
}
WIKI_API = "https://en.wikipedia.org/w/api.php"


DIRECT_CHARS = 8000        # pages shorter than this are returned as-is
CHUNK_CHARS = 6000         # chunk size sent to the local extractor
MAX_OUTPUT_CHARS = 8000    # cap on what the main agent receives

# ---------- HTML -> markdown (lossless) ----------

def _html_to_markdown(html: str, keep_links: bool = True) -> str:
    soup = BeautifulSoup(html, "lxml")
    for selector in [
        "script", "style", "noscript", "nav", "footer", "aside", "form",
        "sup.reference", "span.mw-editsection", "table.navbox", "div.navbox",
        "ol.references", "div.reflist", "div.printfooter",
    ]:
        for el in soup.select(selector):
            el.decompose()
    strip = ["img"] if keep_links else ["img", "a"]
    try:
        text = to_markdown(str(soup.body or soup), heading_style="ATX", strip=strip)
    except RecursionError:
        text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ---------- PDF -> markdown (lossless) ----------
def _pdf_to_markdown(data: bytes) -> str:
    doc = pymupdf.open(stream=data, filetype="pdf")
    return pymupdf4llm.to_markdown(doc)


# ---------- local extractor (gemma3:4b via Ollama) ----------

def _chunk_markdown(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split at line boundaries; repeat the last heading at the top of each new chunk."""
    chunks, current, current_len, heading = [], [], 0, ""
    for line in text.split("\n"):
        if line.startswith("#"):
            heading = line
        if current and current_len + len(line) + 1 > size:
            chunks.append("\n".join(current))
            current = [heading] if heading and not line.startswith("#") else []
            current_len = sum(len(x) + 1 for x in current)
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "between", "what", "which",
    "how", "many", "are", "was", "were", "have", "has", "who", "did", "does", "into",
    "about", "list", "all", "any", "there", "their", "them", "its",
}


def _split_units(text: str, max_chars: int = 4000) -> list[dict]:
    """Split markdown into sections at headings; oversized sections are split into parts."""
    sections, title, buf = [], "(intro)", []
    for line in text.split("\n"):
        if re.match(r"#{1,6} ", line):
            if buf:
                sections.append((title, "\n".join(buf)))
            title, buf = line.lstrip("#").strip(), [line]
        else:
            buf.append(line)
    if buf:
        sections.append((title, "\n".join(buf)))

    units = []
    for title, body in sections:
        if len(body.strip()) < 60:  # heading with no real content
            continue
        parts = _chunk_markdown(body, max_chars)
        for k, part in enumerate(parts, 1):
            name = title if len(parts) == 1 else f"{title} (part {k}/{len(parts)})"
            units.append({"title": name, "text": part})
    return units


def _keywords(question: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    words = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def _score(unit: dict, keys: set[str]) -> int:
    title, body = unit["title"].lower(), unit["text"].lower()
    return sum(3 for k in keys if k in title) + sum(1 for k in keys if k in body)


def _preview(text: str, n: int = 100) -> str:
    body = " ".join(l.strip() for l in text.split("\n") if l.strip() and not l.startswith("#"))
    return body[:n]


def _llm_pick(units: list[dict], question: str) -> list[int]:
    """Ask the local model to SELECT section numbers (1-based). It never writes page text."""
    toc = "\n".join(f"{i}. {u['title']} :: {_preview(u['text'])}" for i, u in enumerate(units, 1))[:12000]
    try:
        resp = litellm.completion(
            model=LOCAL_MODEL,
            api_base=OLLAMA_BASE,
            messages=[{"role": "user", "content": SELECT_PROMPT.format(question=question, toc=toc)}],
            temperature=0,
            num_ctx=8192,
        )
    except Exception:
        return []  # Ollama down: keyword scoring still works
    answer = resp.choices[0].message.content or ""
    nums = [int(n) for n in re.findall(r"\d+", answer)]
    return [n for n in nums if 1 <= n <= len(units)][:4]


def _extract_relevant(text: str, question: str) -> str:
    if len(text) <= DIRECT_CHARS:
        return text

    units = _split_units(text)
    keys = _keywords(question)
    scores = [_score(u, keys) for u in units]
    ranked = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)

    chosen, used = [], 0
    for i in ranked[:4]:
        if scores[i] <= 0:
            break
        size = len(units[i]["text"])
        if used + size <= MAX_OUTPUT_CHARS:
            chosen.append(i)
            used += size

    if not chosen:
        titles = " | ".join(u["title"] for u in units)[:1500]
        return f"No section matched the question. Sections on this page: {titles}"

    body = "\n\n---\n\n".join(units[i]["text"] for i in sorted(chosen))
    others = " | ".join(u["title"] for j, u in enumerate(units) if j not in chosen)[:800]
    return f"{body}\n\n[Other sections on this page: {others}]"
# ---------- fetchers ----------

def _fetch_wikipedia_html(title: str, date: str = "") -> str:
    """Return article HTML, optionally the latest revision on or before `date` (YYYY-MM-DD)."""
    params = {"action": "parse", "prop": "text", "format": "json", "formatversion": 2}
    if date:
        rev = requests.get(
            WIKI_API, headers=HEADERS, timeout=20,
            params={
                "action": "query", "prop": "revisions", "titles": title,
                "rvlimit": 1, "rvdir": "older", "rvstart": f"{date}T23:59:59Z",
                "rvprop": "ids", "redirects": 1, "format": "json", "formatversion": 2,
            },
        )
        rev.raise_for_status()
        page = rev.json()["query"]["pages"][0]
        if page.get("missing") or not page.get("revisions"):
            raise ValueError(f"No revision of '{title}' found on or before {date}.")
        params["oldid"] = page["revisions"][0]["revid"]
    else:
        params["page"] = title
        params["redirects"] = 1

    resp = requests.get(WIKI_API, headers=HEADERS, timeout=30, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"Wikipedia error: {data['error'].get('info', data['error'])}")
    return data["parse"]["text"]


SEARCH_ATTEMPTS = [{}, {"backend": "bing"}, {"backend": "brave"}]


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs and snippets (retries with other backends)."""
    max_results = min(max_results, 6)
    last_error = "No results found."
    for extra in SEARCH_ATTEMPTS:
        try:
            results = DDGS().text(query, max_results=max_results + 4, **extra)
        except Exception as e:
            last_error = str(e) or type(e).__name__
            continue
        results = [r for r in results if "aclick" not in r.get("href", "")][:max_results]
        if results:
            return "\n\n".join(
                f"[{i}] {r['title']}\n    URL: {r['href']}\n    {r['body'][:300]}"
                for i, r in enumerate(results, 1)
            )
    return f"Search failed: {last_error}. Try rephrasing with fewer quotes or different keywords."


def read_webpage(url: str, question: str) -> str:
    """Fetch a web page or PDF and return only the parts relevant to `question`."""
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube pages cannot be read with this tool. Videos need a dedicated video tool; do not retry this URL."
    try:
        resp = requests.get(url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        return f"Failed to fetch page: {e}"

    is_pdf = "pdf" in resp.headers.get("Content-Type", "").lower() or url.lower().split("?")[0].endswith(".pdf")
    try:
        text = _pdf_to_markdown(resp.content) if is_pdf else _html_to_markdown(resp.text, keep_links=True)
    except Exception as e:
        return f"Failed to parse content: {type(e).__name__}: {e}"
    return _extract_relevant(text, question)


def read_wikipedia(title: str, question: str, date: str = "") -> str:
    """Read an English Wikipedia article (optionally as of `date`) and return the parts relevant to `question`."""
    try:
        html = _fetch_wikipedia_html(title, date)
    except Exception as e:
        return f"Failed to fetch Wikipedia page: {e}"
    return _extract_relevant(_html_to_markdown(html, keep_links=False), question)


# ---------- tool registration ----------

WEB_TOOLS = [
    _tool(
        "web_search",
        "Search the web. Returns titles, URLs and short snippets. Use read_webpage afterwards to read a promising result.",
        {
            "query": {"type": "string", "description": "The search query."},
            "max_results": {"type": "integer", "description": "Number of results (default 5)."},
        },
        ["query"],
        web_search,
    ),
    _tool(
        "read_webpage",
        "Read a web page by URL. Returns only the passages relevant to `question` (copied verbatim, tables included). "
        "If the result looks incomplete, call again with a differently worded question.",
        {
            "url": {"type": "string", "description": "Full URL starting with https://"},
            "question": {"type": "string", "description": "Specific description of the information you need from this page."},
        },
        ["url", "question"],
        read_webpage,
    ),
    _tool(
        "read_wikipedia",
        "Read an English Wikipedia article by exact title. Returns only the passages relevant to `question`. "
        "Pass `date` (YYYY-MM-DD) to read the article as it was on that date, e.g. when the question mentions a specific year's version.",
        {
            "title": {"type": "string", "description": "Exact article title, e.g. 'Mercedes Sosa'."},
            "question": {"type": "string", "description": "Specific description of the information you need, e.g. 'studio albums with release years'."},
            "date": {"type": "string", "description": "Optional YYYY-MM-DD."},
        },
        ["title", "question"],
        read_wikipedia,
    ),
]