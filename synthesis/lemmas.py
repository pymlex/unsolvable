"""Thematic lemma sampling via Russian word embeddings (navec).

Protocol:
1. Exclude function words and top-frequency lemmas.
2. Sample a seed lemma with enough udarenieru senses.
3. Take mid-similarity embedding neighbors (related, not duplicates, not random).
4. Keep only neighbors that are also polysemous in udarenieru.

Embedding source: Natasha navec ``hudlit_v1_12B_500K_300d_100q``.
Similarity band: cosine in ``[SIM_LOW, SIM_HIGH]`` (default 0.28–0.72).
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path


FUNCTION_WORDS = {
    "а", "без", "близ", "в", "вместо", "вне", "для", "до", "за", "из", "из-за",
    "из-под", "к", "кроме", "между", "на", "над", "о", "об", "обо", "около",
    "от", "перед", "по", "под", "после", "при", "про", "ради", "с", "со",
    "сквозь", "среди", "у", "через", "и", "или", "либо", "ни", "но", "однако",
    "зато", "тоже", "также", "да", "чтобы", "чтоб", "если", "хотя", "хоть",
    "когда", "пока", "как", "будто", "словно", "точно", "чем", "что", "ли",
    "же", "бы", "б", "не", "вот", "вон", "даже", "уже", "ещё", "еще", "ведь",
    "лишь", "только", "именно", "почти", "разве", "неужели", "пусть", "пускай",
    "якобы", "дескать", "мол", "это", "то", "все", "всё", "сам", "сама", "само",
    "сами", "мой", "твой", "наш", "ваш", "его", "её", "ее", "их", "свой", "я",
    "ты", "он", "она", "оно", "мы", "вы", "они", "кто", "какой", "какая",
    "какое", "какие", "чей", "чья", "чьё", "чье", "который", "которая",
    "которое", "которые", "где", "куда", "откуда", "почему", "зачем", "сколько",
    "этот", "эта", "эти", "тот", "та", "те", "такой", "такая", "такое", "такие",
    "один", "одна", "одно", "два", "две", "три", "четыре", "пять", "шесть",
    "семь", "восемь", "девять", "десять", "сто", "тысяча", "миллион", "первый",
    "второй", "третий", "быть", "есть", "был", "была", "было", "были", "будет",
}

POLYSEME_SEEDS = [
    "замок", "коса", "лук", "мир", "ключ", "язык", "ручка", "свет", "труба",
    "газ", "орган", "лист", "банк", "среда", "вид", "долг", "пол", "брак",
    "титан", "голос", "курс", "ряд", "состав", "удар", "ход", "часть", "сила",
    "поле", "стол", "бокс", "бор", "вар", "винт", "гладь", "град", "гриф",
    "завод", "зуб", "икра", "капот", "карта", "каток", "клетка", "кошка",
    "край", "кредит", "круг", "крыло", "лавка", "ласка", "лента", "лира",
    "лом", "марка", "масло", "мат", "мель", "мера", "метка", "мина", "мода",
    "молния", "мост", "мышь", "наряд", "нос", "нота", "образ", "операция",
    "орел", "ось", "отпуск", "пар", "партия", "перо", "печать", "пика",
    "плата", "плод", "пост", "пояс", "провод", "продукт", "проспект", "пыл",
    "рама", "ранг", "редакция", "рок", "рост", "рубеж", "сад", "свеча",
    "сеть", "скат", "скот", "слой", "сок", "сплав", "справка", "ставка",
    "стан", "ствол", "степень", "стиль", "сторона", "струна", "сутки",
    "сфера", "сцена", "таз", "такт", "ток", "точка", "трава", "тракт", "тур",
    "тяга", "уклон", "устав", "участок", "фабрика", "фигура", "фонд", "форма",
    "фраза", "фронт", "характер", "хвост", "царапина", "цепь", "чаша",
    "червь", "шахта", "шина", "шкала", "штаб", "шум", "щит", "экспедиция",
    "элемент", "ядро", "яйцо", "яма", "ярус", "журнал", "газета", "статья",
    "репортаж", "пациент", "врач", "хирург", "диагноз", "симптом", "терапия",
    "клиника", "аптека", "доза", "препарат", "анализ", "лаборатория",
    "исследование", "эксперимент", "гипотеза", "теория", "метод", "процесс",
    "система", "структура", "механизм", "аппарат", "инструмент", "материал",
    "металл", "деталь", "узел", "схема", "чертеж", "проект", "смета",
    "подряд", "контракт", "договор", "акт", "протокол", "архив", "документ",
    "отчет", "сводка", "брифинг",
]

SIM_LOW = 0.28
SIM_HIGH = 0.72
MIN_SENSES = 2
FREQ_CANDIDATE_START = 2000
FREQ_CANDIDATE_END = 12000

EMBEDDING_SOURCE = (
    "navec hudlit_v1_12B_500K_300d_100q "
    "(Natasha; data/navec_hudlit_v1_12B_500K_300d_100q.tar)"
)
SIMILARITY_BAND_DOC = f"cosine mid-band [{SIM_LOW}, {SIM_HIGH}]"


def _repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_top_frequent_lemmas(n: int = 2000) -> set[str]:
    """Load top-N lemmas from an open Russian frequency list cached under data/."""
    path = _repo_data_dir() / "ru_freq_top2000.txt"
    if not path.exists():
        src = _repo_data_dir() / "ru_freq_50k.txt"
        if src.exists():
            lines = []
            for line in src.read_text(encoding="utf-8").splitlines():
                token = line.strip().split()[0].lower() if line.strip() else ""
                if token:
                    lines.append(token)
                if len(lines) >= n:
                    break
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_frequency_ranked() -> list[str]:
    """Full ranked lemma list from the cached FrequencyWords dump."""
    src = _repo_data_dir() / "ru_freq_50k.txt"
    if not src.exists():
        return []
    out: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0].lower() if line.strip() else ""
        if token:
            out.append(token)
    return out


def is_excluded_lemma(lemma: str, top_frequent: set[str] | None = None) -> bool:
    """Return True if lemma is a function word or ultra-frequent content word."""
    low = lemma.strip().lower()
    if not low or " " in low:
        return True
    if low in FUNCTION_WORDS:
        return True
    freq = top_frequent if top_frequent is not None else load_top_frequent_lemmas()
    return low in freq


def eligible_lemma_pool(top_n: int = 2000) -> list[str]:
    """Seed lemmas minus function words and top-frequency lemmas."""
    top = load_top_frequent_lemmas(top_n)
    out: list[str] = []
    seen: set[str] = set()
    for lemma in POLYSEME_SEEDS:
        low = lemma.lower()
        if low in seen or is_excluded_lemma(low, top):
            continue
        seen.add(low)
        out.append(low)
    return out


@lru_cache(maxsize=1)
def load_navec():
    """Load the cached Natasha navec pack from data/."""
    from navec import Navec

    path = _repo_data_dir() / "navec_hudlit_v1_12B_500K_300d_100q.tar"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing navec pack at {path}. Expected {EMBEDDING_SOURCE}"
        )
    return Navec.load(str(path))


@lru_cache(maxsize=1)
def _candidate_matrix():
    """Vectors for mid-frequency + seed candidates used in neighbor search."""
    import numpy as np

    nav = load_navec()
    top = load_top_frequent_lemmas()
    ranked = load_frequency_ranked()
    band = ranked[FREQ_CANDIDATE_START:FREQ_CANDIDATE_END]
    words: list[str] = []
    seen: set[str] = set()
    for w in list(eligible_lemma_pool()) + band:
        if w in seen or w not in nav:
            continue
        if is_excluded_lemma(w, top):
            continue
        if not w.isalpha() or len(w) < 3:
            continue
        seen.add(w)
        words.append(w)
    mat = np.stack([np.asarray(nav[w], dtype=np.float32) for w in words])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return words, mat


@lru_cache(maxsize=8192)
def sense_count(lemma: str) -> int:
    """Cached udarenieru sense count for polysemy gating."""
    from synthesis.dictionary import fetch_senses

    return len(fetch_senses(lemma, max_senses=32))


def mid_similarity_neighbors(
    seed: str,
    *,
    sim_low: float = SIM_LOW,
    sim_high: float = SIM_HIGH,
    limit: int = 60,
) -> list[tuple[str, float]]:
    """Return embedding neighbors in the mid similarity band for ``seed``."""
    import numpy as np

    nav = load_navec()
    if seed not in nav:
        return []
    words, mat = _candidate_matrix()
    vec = np.asarray(nav[seed], dtype=np.float32)
    denom = float(np.linalg.norm(vec))
    if denom <= 0:
        return []
    vec = vec / denom
    sims = mat @ vec
    scored: list[tuple[str, float]] = []
    mid = 0.5 * (sim_low + sim_high)
    for word, sim in zip(words, sims):
        s = float(sim)
        if word == seed:
            continue
        if sim_low <= s <= sim_high:
            scored.append((word, s))
    scored.sort(key=lambda x: abs(mid - x[1]))
    return scored[:limit]


def sample_thematic_lemmas(
    n: int = 5,
    rng: random.Random | None = None,
    *,
    sim_low: float = SIM_LOW,
    sim_high: float = SIM_HIGH,
    min_senses: int = MIN_SENSES,
) -> list[str]:
    """Sample ``n`` thematically related polysemous lemmas around one seed."""
    rng = rng or random.Random()
    nav = load_navec()
    seeds = [w for w in eligible_lemma_pool() if w in nav]
    if not seeds:
        raise RuntimeError("no eligible seeds present in navec vocabulary")

    for _ in range(50):
        seed = rng.choice(seeds)
        if sense_count(seed) < min_senses:
            continue
        neighbors = mid_similarity_neighbors(
            seed, sim_low=sim_low, sim_high=sim_high
        )
        rng.shuffle(neighbors)
        picked = [seed]
        for word, _sim in neighbors:
            if word in picked:
                continue
            if sense_count(word) < min_senses:
                continue
            picked.append(word)
            if len(picked) >= n:
                return picked[:n]
    raise RuntimeError("failed to sample thematic polysemous lemma set")


def sample_lemmas(n: int = 5, rng: random.Random | None = None) -> list[str]:
    """Public entry used by synthesis: thematic mid-similarity sampling."""
    return sample_thematic_lemmas(n=n, rng=rng)
