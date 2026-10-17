"""Lemma pools for synthetic tip-2 generation."""

from __future__ import annotations

import random


DEFAULT_LEMMAS = [
    "замок",
    "коса",
    "лук",
    "мир",
    "ключ",
    "язык",
    "лисица",
    "ручка",
    "свет",
    "труба",
    "газ",
    "орган",
    "лист",
    "банк",
    "среда",
    "вид",
    "долг",
    "пол",
    "брак",
    "титан",
    "голос",
    "курс",
    "ряд",
    "состав",
    "удар",
    "ход",
    "часть",
    "сила",
    "поле",
    "стол",
]


def sample_lemmas(n: int = 5, rng: random.Random | None = None) -> list[str]:
    """Sample ``n`` distinct lemmas from the default pool."""
    rng = rng or random.Random()
    return rng.sample(DEFAULT_LEMMAS, k=min(n, len(DEFAULT_LEMMAS)))
