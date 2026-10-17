# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

## 1. Цель

Нужны тестовые задания на понимание значения слова в тексте. Человек решает их по смыслу. **SLM**, small language model, малая языковая модель, ошибается из-за поверхностных эвристик.

Рабочий прогон идёт на заданиях ЕГЭ по русскому **типа 2**: текст, пять выделенных слов, пять толкований, ответ — склейка номеров верных вариантов, например `35`. Датасет для нулевой базы: [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026).

Две цели нельзя смешивать без контроля. **Construct validity** — соответствие задания заявленному навыку человека. Anti-SLM objective — заставить модель ошибиться. Совпадение этих целей доказывают эмпирически на батареи моделей при неизменном ключе.

В проекте **нет** human pilot и **нет** цикла валидации людьми. Приёмка item опирается на автоматические ограничения и на батарее SLM при температуре 0.

## 2. Печатное ограничение

Задание должно оставаться корректным для печатного экзамена. Допустимы правки на уровне символов и порядка слов **только если** текст остаётся грамматически и пунктуационно правильным русским. Unicode-lookalike, невидимые символы, опечатки ради атаки и любая порча, из-за которой текст нельзя напечатать в КИМ, запрещены.

Критерий выпуска: человек со школьной подготовкой читает item как нормальное задание с тем же ключом. Модель при этом ошибается чаще, чем на исходной формулировке.

**SLM battery** — фиксированный набор малых AR-моделей, на которых одновременно проверяют трудность. Так не подгоняют item под одну модель. Декодирование только greedy или temperature 0.

## 3. Как модель видит задачу

**MCQ** — multiple-choice question: выбор среди фиксированных опций. Одна единица теста — **item**: контекст, целевые слова, опции-толкования. Неверные правдоподобные опции — **distractors**.

Значение многозначного слова — **sense**. Выбор верного sense — близко к **WSD**, word sense disambiguation, и к **WiC**, Word-in-Context. Здесь формат экзаменационный: несколько верных номеров из пяти толкований.

Малая **AR-модель**, autoregressive language model, оценивает вероятность ответа после промпта. Пусть контекст $c$, опции $O$, ключ $y^\star$. Модель даёт

$$
\hat{y} = \arg\max_k\, p_\theta\left(a_k \mid \mathrm{prompt}(c, O)\right).
$$

Важные смещения: **option-order bias** и **selection bias** на метки опций, **shortcut** по пересечению слов, **collocation** prior по устойчивым соседствам. На русском у малых моделей collocational prior особенно силён.

## 4. Вердикт

Идея опирается на известные факты: позиционный bias MCQ, shortcut на артефактах, дискретная adversarial-оптимизация текста, AFLite-фильтры лёгких примеров. Новизна не в отдельных мотивах.

Рабочий зазор проекта — пайплайн под русский конструкт «значение слова в контексте» с прогоном на ЕГЭ типа 2, без human-пилота, с жёстким печатным ограничением и с батареей малых моделей при temperature 0. Вклад возможен как композиция и адаптация, а не как переоткрытие блоков.

## 5. Что уже есть в литературе

Кратко. Zheng et al. (ICLR 2024) — selection bias и PriDe. Pezeshkpour и Hruschka — разрыв accuracy при перестановке опций. Gururangan et al. — annotation artifacts. McCoy et al. — HANS. Sakaguchi et al. и Le Bras et al. — AFLite. Ebrahimi et al. — HotFlip. Morris et al. — TextAttack. GradQuiz и смежные работы — adversarial distractors в образовании. Jain и Wallace, Wiegreffe и Pinter — осторожность с attention как объяснением. Полные ссылки в конце файла.

Прямой перенос опечаток и Unicode-атак разрушает экзаменационный конструкт. Их не используем.

## 6. Вне скоупа

Следующее **вне скоупа** текущего плана, потому что требует human raters или пилота на людях:

- human validity panel и двойная экспертная разметка agreement;
- IRT-гейт по оценённой на людях трудности $b_j$;
- пилот на школьниках для измерения human accuracy $H$;
- DIF по демографиям.

Эти блоки можно вернуть позже. Сейчас их нет в пайплайне выпуска.

## 7. Методы фреймворка

Пайплайн: генерация кандидатов, печатный grammatical filter, adversarial hardening на battery при temp 0, surface-filter, zero-edit и hardened eval. Ключ $y^\star$ при hardening **не меняют**, если не оговорено иное.

### M1. Sense-frequency traps

Берут слово $w$ с несколькими senses. Контекст пишут так, чтобы для человека однозначно читался целевой sense $s^\star$, а более частотный sense оставался лексически соблазнительным для LM. Опции: верное толкование, толкование частотного sense, словарьно допустимые ближние distractors.

Ловушка бьёт collocational и frequency prior малой модели. Текст остаётся нормальным русским. Измерение: падение exact-match на battery при неизменном ключе относительно исходного пула.

### M2. Grammatical paraphrase контекста

Парафраз $c \to c'$ сохраняет смысл для человека и пунктуационную норму. Меняют порядок слов, синонимы в пределах нормы, синтаксис. Цель — сбить collocational cues и lexical overlap, на которые опирается SLM, не ломая читаемость.

Поиск дискретный: кандидаты парафраза фильтруют грамматическим/языковым чекером или сильной LM-as-judge **только** на норму и сохранение смысла ключа, без human panel. Кандидат принимают, если battery accuracy падает, а ключ по правилам задания тот же.

### M3. Option-order permutation как eval probe

Перестановки опций **не** публикуют как новый ключ бланка. Это зонд: измеряют, держится ли ошибка модели при $\pi(O)$. Item считают model-hard по смыслу только если ошибка устойчива на нескольких перестановках. Иначе трудность может быть артефактом метки.

Формально

$$
\mathrm{Consist}(x) = \frac{1}{R} \sum_{r=1}^{R} \mathbf{1}\left[\hat{y}\left(\pi_r(O)\right) = y^\star\right].
$$

Низкая Consist к верному ответу при temp 0 на battery — сигнал смысловой неустойчивости модели, а не только order bias.

### M4. Distractor rewriting в словарьных границах

Фиксируют $c$, $w$, $y^\star$. Переписывают неверные опции так, чтобы они оставались словарно и грамматически валидными толкованиями, но сильнее тянули LM prior к частотному или соседнему sense. Запрещены бессмысленные строки, опечатки, Unicode-подмены.

Цель поиска:

$$
\max_{d} \sum_{m \in \mathrm{battery}} \mathbf{1}\left[\hat{y}_m(c, O[d]) \neq y^\star\right]
$$

при условии печатной корректности $d$. Transfer проверяют на held-out модели вне battery.

### M5. Discrete search грамматических near-paraphrases

Локальный поиск по окрестности item: одна правка за шаг из разрешённого множества трансформаций. Разрешены только правки, после которых текст остаётся грамматически и пунктуационно корректным. Цель — минимизировать accuracy battery при неизменном $y^\star$.

Это TextAttack-подобная схема с жёстким constraint «printable exam Russian», без character-noise goal. Остановка по бюджету шагов или по плато метрики.

### M6. AFLite-like surface filter на SLM battery при temp 0

Строят слабые решатели и урезанные промпты: только опции, только целевые слова, bag-of-words overlap с верной опцией. Items, которые weak path или слабая модель стабильно решает при temp 0 без полного контекста, помечают как surface-solvable и отфильтровывают из hard slice.

В отличие от классического AFLite с линейным классификатором на эмбеддингах, здесь основной фильтр — сама battery SLM на урезанных входах. Human agreement в приёмку не входит.

### Композиция

M1 даёт пул с контролируемым sense. M2 и M5 правят контекст. M4 правит distractors. M3 проверяет, что трудность не сводится к порядку опций. M6 убирает surface-solvable хвост. Eval всегда включает zero-edit прогон всего набора при temperature 0.

## 8. Метрики

**Exact-match accuracy $M$** — доля items, где нормализованный ответ модели совпал с ключом. Ответ нормализуют до множества цифр 1–5 в каноническом порядке, как в бланке типа 2.

**95% CI** — Wilson score interval и bootstrap по items. Так оценивают значимость на полном наборе без правок и на hard slice после hardening.

**Shuffle Consist** — устойчивость к перестановке опций. Нужна как диагностика, не как замена exact-match.

**Options-only / context-ablated accuracy** — утечка смысла в опции или игнор контекста.

**ASR** — доля items, где hardening заставило модель ошибиться относительно zero-edit базы. Без печатного фильтра ASR не интерпретируют.

**Weak-solvable rate** — доля items, решаемых урезанным входом или слабым путём. После M6 показатель должен падать.

**Transfer** — те же items на модели вне battery.

Метрики human accuracy и $\kappa$ agreement в текущем плане не собирают. Их отсутствие ограничивает claim: утверждают устойчивость к SLM при печатной норме, а не психометрическую валидность для людей.

## 9. Протокол eval

### 9.1. Модели

По умолчанию Qwen2.5-класс малого размера, если ресурс позволяет. Иначе тот же скрипт с подставляемой моделью. Held-out — одна модель другой семьи близкого размера.

### 9.2. Декодирование

Только temperature 0 / `do_sample=False`. Один фиксированный промпт с требованием цифр в формате ключа. Без few-shot с утечкой того же слова.

### 9.3. Базы

1. Zero-edit: все items датасета без правок.
2. После hardening: тот же ключ, изменённые контекст или distractors в печатной норме.
3. Order probe: перестановки опций.
4. Ablation: options-only и context-ablated.

### 9.4. MVP без людей

1. Зафиксировать zero-edit baseline на полном [датасете tip2](https://huggingface.co/datasets/pymlex/reshuege-ru-2026).
2. На items с высоким $M$ применить M2/M4/M5 под печатным constraint.
3. Прогнать M3 и M6.
4. Сравнить $M$, CI, Consist, weak-solvable rate, transfer.

Успех MVP: на hard slice средний $M$ battery ниже zero-edit базы, Consist к ключу не объясняется одной перестановкой, текст остаётся грамматически нормальным, ключ неизменен.

## 10. Риски

**Construct validity.** Без human-пилота нельзя утверждать школьную валидность. Claim ограничивают анти-SLM устойчивостью при печатной норме.

**Fairness.** Слишком редкий sense или узкий контекст могут ударить по разным группам учеников сильнее, чем по модели. Автоматический фильтр частоты sense снижает риск, но не заменяет DIF.

**Dual-use.** Не публиковать генераторы невидимых Unicode-perturbations и ключи закрытых КИМ.

**Нестабильность hard set.** Battery и периодический refresh обязательны.

## BibTeX

```bibtex
@inproceedings{zheng2024mcq,
  title={Large Language Models Are Not Robust Multiple Choice Selectors},
  author={Zheng, Chujie and Zhou, Hao and Meng, Fandong and Zhou, Jie and Huang, Minlie},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

@inproceedings{pezeshkpour2024order,
  title={Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions},
  author={Pezeshkpour, Pouya and Hruschka, Estevam},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2024},
  year={2024}
}

@inproceedings{gururangan2018artifacts,
  title={Annotation Artifacts in Natural Language Inference Data},
  author={Gururangan, Suchin and Swayamdipta, Swabha and Levy, Omer and Schwartz, Roy and Bowman, Samuel and Smith, Noah A.},
  booktitle={Proceedings of NAACL-HLT},
  year={2018}
}

@inproceedings{mccoy2019hans,
  title={Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference},
  author={McCoy, R. Thomas and Pavlick, Ellie and Linzen, Tal},
  booktitle={Proceedings of ACL},
  year={2019}
}

@inproceedings{sakaguchi2020winogrande,
  title={{WinoGrande}: An Adversarial {Winograd} Schema Challenge at Scale},
  author={Sakaguchi, Keisuke and Le Bras, Ronan and Bhagavatula, Chandra and Choi, Yejin},
  booktitle={Proceedings of AAAI},
  year={2020}
}

@inproceedings{lebras2020aflite,
  title={Adversarial Filters of Dataset Biases},
  author={Le Bras, Ronan and Swayamdipta, Swabha and Bhagavatula, Chandra and Zellers, Rowan and Peters, Matthew and Sabharwal, Ashish and Choi, Yejin},
  booktitle={Proceedings of ICML},
  year={2020}
}

@inproceedings{ebrahimi2018hotflip,
  title={{HotFlip}: White-Box Adversarial Examples for Text Classification},
  author={Ebrahimi, Javid and Rao, Anyi and Lowd, Daniel and Dou, Dejing},
  booktitle={Proceedings of ACL (Short Papers)},
  year={2018}
}

@inproceedings{morris2020textattack,
  title={{TextAttack}: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in {NLP}},
  author={Morris, John and Lifland, Eli and Yoo, Jin Yong and Grigsby, Jake and Jin, Di and Qi, Yanjun},
  booktitle={Proceedings of EMNLP: System Demonstrations},
  year={2020}
}

@inproceedings{jain2019attention,
  title={Attention is not Explanation},
  author={Jain, Sarthak and Wallace, Byron C.},
  booktitle={Proceedings of NAACL-HLT},
  year={2019}
}

@misc{unsolvable-2026,
  title={A Framework for Generating Adversarial Items Robust to {SLM} Solvers},
  author={{Фреймворк генерации состязательных заданий, устойчивых к решению SLM}},
  howpublished={https://github.com/pymlex/unsolvable},
  year={2026}
}

@misc{reshuege-ru-2026,
  title={reshuege-ru-2026: {EGE} Russian Type 2 items from Reshu {EGE}},
  author={{pymlex}},
  howpublished={https://huggingface.co/datasets/pymlex/reshuege-ru-2026},
  year={2026}
}
```

The project is under GPL-3.0 license.
