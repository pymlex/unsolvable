"""Lemma pools for synthetic tip-2 generation with frequency filters."""

from __future__ import annotations

import random
from pathlib import Path


FUNCTION_WORDS = {
    "а", "без", "близ", "в", "вместо", "вне", "для", "до", "за", "из", "из-за",
    "из-под", "к", "кроме", "между", "на", "над", "о", "об", "обо", "около",
    "от", "перед", "по", "под", "после", "при", "про", "ради", "с", "со",
    "сквозь", "среди", "у", "через", "и", "или", "либо", "ни", "но", "однако",
    "зато", "тоже", "также", "да", "чтобы", "чтоб", "если", "хотя", "хоть",
    "когда", "пока", "как", "будто", "словно", "точно", "чем", "что", "чтобы",
    "ли", "же", "бы", "б", "не", "ни", "вот", "вон", "даже", "уже", "ещё",
    "еще", "ведь", "лишь", "только", "именно", "почти", "разве", "неужели",
    "пусть", "пускай", "якобы", "дескать", "мол", "это", "то", "все", "всё",
    "сам", "сама", "само", "сами", "мой", "твой", "наш", "ваш", "его", "её",
    "ее", "их", "свой", "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "кто", "что", "какой", "какая", "какое", "какие", "чей", "чья", "чьё",
    "чье", "который", "которая", "которое", "которые", "где", "куда", "откуда",
    "когда", "почему", "зачем", "сколько", "этот", "эта", "это", "эти", "тот",
    "та", "те", "такой", "такая", "такое", "такие", "один", "одна", "одно",
    "два", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
    "десять", "сто", "тысяча", "миллион", "первый", "второй", "третий",
    "быть", "есть", "был", "была", "было", "были", "будет", "являться",
}

POLYSEME_SEEDS = [
    "замок", "коса", "лук", "мир", "ключ", "язык", "ручка", "свет", "труба",
    "газ", "орган", "лист", "банк", "среда", "вид", "долг", "пол", "брак",
    "титан", "голос", "курс", "ряд", "состав", "удар", "ход", "часть", "сила",
    "поле", "стол", "бокс", "бор", "вар", "винт", "гард", "гладь", "град",
    "гриф", "губы", "добыча", "жалоба", "завод", "зуб", "икра", "капот",
    "карта", "каток", "класс", "клетка", "кнут", "косая", "кошка", "край",
    "кредит", "круг", "крыло", "лавка", "ласка", "лента", "лира", "лом",
    "магазин", "марка", "масло", "мат", "мель", "мера", "метка", "мина",
    "мода", "молния", "мост", "мотивация", "мышь", "наряд", "нос", "нота",
    "образ", "окно", "операция", "орел", "ось", "отпуск", "пар", "партия",
    "перо", "печать", "пика", "план", "плата", "плод", "пост", "пояс",
    "право", "провод", "продукт", "проспект", "путь", "пыл", "рама", "ранг",
    "редакция", "рок", "рост", "рубеж", "ряд", "сад", "свеча", "семейство",
    "сеть", "скат", "скот", "слой", "совет", "сок", "состав", "сплав",
    "справка", "ставка", "стан", "ствол", "степень", "стиль", "сторона",
    "струна", "суд", "сутки", "сфера", "сцена", "таз", "такт", "тока",
    "ток", "точка", "трава", "тракт", "тур", "тяга", "уклон", "устав",
    "участок", "фабрика", "фигура", "фонд", "форма", "фраза", "фронт",
    "характер", "хвост", "ходка", "царапина", "центр", "цепь", "чаша",
    "червь", "шаг", "шахта", "шина", "шкала", "штаб", "шум", "щит",
    "экспедиция", "элемент", "ядро", "яйцо", "яма", "ярус",
]


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
    """Polyseme seeds minus function words and top-frequency lemmas."""
    top = load_top_frequent_lemmas(top_n)
    out: list[str] = []
    seen: set[str] = set()
    for lemma in POLYSEME_SEEDS:
        low = lemma.lower()
        if low in seen:
            continue
        if is_excluded_lemma(low, top):
            continue
        seen.add(low)
        out.append(low)
    return out


def sample_lemmas(n: int = 5, rng: random.Random | None = None) -> list[str]:
    """Sample ``n`` distinct eligible lemmas for one tip-2 item."""
    rng = rng or random.Random()
    pool = eligible_lemma_pool()
    if len(pool) < n:
        raise RuntimeError(f"eligible lemma pool too small: {len(pool)} < {n}")
    return rng.sample(pool, k=n)
