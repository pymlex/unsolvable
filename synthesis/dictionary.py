"""Dictionary sense fetch for synthetic tip-2 generation.

Primary source: udarenieru.ru HTML pages (Evgenyeva glosses).
There is no public JSON API; senses are scraped from
``index.php?word=on&mas_word=<lemma>`` (Толкование).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html import unescape
from functools import lru_cache


UDARENIE_RU_BASE = "https://udarenieru.ru/index.php"
UDARENIE_RU_DOC = (
    "udarenieru.ru has no public JSON API. Senses are scraped from HTML "
    "Толкование pages: GET https://udarenieru.ru/index.php?word=on&mas_word=<lemma> "
    "(Evgenyeva A.P. glosses inside itemprop=text / numbered <p><b>N.</b> blocks "
    "and inline first senses after POS markers)."
)
USER_AGENT = "unsolvable/1.0 (+https://github.com/pymlex/unsolvable)"


def fetch_udarenieru_html(lemma: str, timeout_sec: float = 30.0) -> str:
    """Download the udarenieru.ru толкование HTML page for one lemma."""
    query = urllib.parse.urlencode({"word": "on", "mas_word": lemma})
    url = f"{UDARENIE_RU_BASE}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return response.read().decode("utf-8", errors="replace")


def _strip_tags(chunk: str) -> str:
    chunk = re.sub(r"<i>[\s\S]*?</i>", " ", chunk)
    chunk = re.sub(r"<em>[\s\S]*?</em>", " ", chunk)
    chunk = re.sub(r"<br\s*/?>", " ", chunk, flags=re.I)
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    return chunk


def _clean_gloss(raw: str) -> str:
    body = _strip_tags(raw)
    body = body.replace("□", " ").replace("||", ". ")
    body = re.sub(r"\s+", " ", body).strip()
    body = re.sub(
        r"^(Устар\.|Воен\.|Тех\.|Архит\.|Спец\.|разг\.|прост\.|книжн\.|перен\.;?)\s*",
        "",
        body,
        flags=re.I,
    )
    body = re.sub(r"^\([^)]{0,40}\)\s*", "", body)
    body = re.sub(r"^или\s+", "", body, flags=re.I)
    body = re.split(r"\s+[□◊]", body)[0]
    body = re.split(
        r"(?<=[.!?])\s+(?:[А-ЯЁ]\.\s*)?[А-ЯЁ][а-яё\-]+(?:\s+[А-ЯЁ]\.)*(?:\s+[А-ЯЁ][а-яё\-]+)*,",
        body,
    )[0]
    body = re.sub(r"\s+", " ", body).strip(" .;")
    body = re.sub(r"(?:\s+[А-ЯЁA-Z]{1,3})+$", "", body).strip(" .;")
    if len(body) > 220:
        cut = body.find(". ")
        if cut >= 20:
            body = body[:cut].strip()
    return body


def extract_senses_from_udarenieru_html(html: str) -> list[str]:
    """Extract literal gloss strings from an udarenieru толкование page."""
    html = unescape(html)
    html = re.sub(r"&#x301;|\u0301", "", html)
    chunks = re.findall(r'itemprop="text"[^>]*>([\s\S]*?)</div>', html)
    if not chunks:
        m = re.search(r'id="article_container"[\s\S]*', html)
        chunks = [m.group(0)] if m else [html]

    senses: list[str] = []
    for block in chunks:
        for para in re.findall(r"<p>([\s\S]*?)</p>", block):
            plain_head = _strip_tags(para)
            if plain_head.strip().startswith(("◊", "◇")) or plain_head.strip().startswith("&loz;"):
                continue
            for m in re.finditer(r"<b>\s*(\d+)\.\s*</b>\s*([\s\S]*)", para):
                gloss = _clean_gloss(m.group(2))
                if len(gloss) >= 12 and gloss not in senses:
                    senses.append(gloss)
            if re.search(r"<b>\s*\d+\.\s*</b>", para):
                continue
            m_inline = re.search(
                r"(?:^|>)\s*(?:[мжс]\.|ср\.|мн\.)\s*([\s\S]+)$",
                plain_head,
                flags=re.I,
            )
            if not m_inline:
                m_inline = re.search(
                    r",\s*(?:м|ж|ср|мн)\.\s+([А-ЯЁа-яёA-Za-z].+)$",
                    plain_head,
                )
            if m_inline:
                gloss = _clean_gloss(m_inline.group(1))
                if len(gloss) >= 12 and gloss not in senses:
                    senses.append(gloss)
            else:
                for m in re.finditer(r"(?:^|\s)(\d+)\.\s+([А-ЯЁа-яёA-Za-z][^.]{8,200})", plain_head):
                    gloss = _clean_gloss(m.group(2))
                    if len(gloss) >= 12 and gloss not in senses:
                        senses.append(gloss)
    return senses


@lru_cache(maxsize=4096)
def fetch_senses_udarenieru(lemma: str, max_senses: int = 32) -> tuple[str, ...]:
    """Fetch and parse all dictionary senses for one Russian lemma from udarenieru.ru."""
    html = fetch_udarenieru_html(lemma)
    senses = extract_senses_from_udarenieru_html(html)
    return tuple(senses[:max_senses])


def fetch_senses(lemma: str, max_senses: int = 32) -> list[str]:
    """Public entry: udarenieru.ru senses for ``lemma``."""
    return list(fetch_senses_udarenieru(lemma, max_senses=max_senses))
