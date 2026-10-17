"""Dictionary sense fetch for synthetic tip-2 generation.

Primary source: Russian Wiktionary MediaWiki API (no key required).
udarenieru-style endpoints are documented as optional when available.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request


WIKTIONARY_API = "https://ru.wiktionary.org/w/api.php"
UDARENIE_RU_DOC = (
    "udarenieru.ru has no stable public JSON sense API in this environment. "
    "Use Wiktionary via fetch_senses_wiktionary, or plug a paid dictionary later."
)


def fetch_wikitext(lemma: str, timeout_sec: float = 30.0) -> str:
    """Download Russian Wiktionary wikitext for one lemma."""
    query = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": lemma,
            "prop": "wikitext",
            "format": "json",
        }
    )
    url = f"{WIKTIONARY_API}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "unsolvable/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["parse"]["wikitext"]["*"]


def extract_senses_from_wikitext(wikitext: str, max_senses: int = 8) -> list[str]:
    """Extract numbered definition lines from Wiktionary wikitext."""
    senses: list[str] = []
    for line in wikitext.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#") or stripped.startswith("##"):
            continue
        if stripped.startswith("#:"):
            continue
        body = stripped.lstrip("#").strip()
        body = re.sub(r"\{\{[^}]+\}\}", " ", body)
        body = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", body)
        body = re.sub(r"'{2,}", "", body)
        body = re.sub(r"\s+", " ", body).strip(" .;")
        if len(body) < 8:
            continue
        senses.append(body)
        if len(senses) >= max_senses:
            break
    return senses


def fetch_senses_wiktionary(lemma: str, max_senses: int = 8) -> list[str]:
    """Fetch and parse dictionary senses for one Russian lemma."""
    wikitext = fetch_wikitext(lemma)
    return extract_senses_from_wikitext(wikitext, max_senses=max_senses)


def fetch_senses(lemma: str, max_senses: int = 8) -> list[str]:
    """Public entry: Wiktionary senses for ``lemma``."""
    return fetch_senses_wiktionary(lemma, max_senses=max_senses)
