# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

## 1. Постановка

Нужны задания на значение слова в контексте. Человек решает их по смыслу. SLM ошибается на поверхностных эвристиках, collocational prior и порядке опций.

Рабочий прогон идёт на заданиях ЕГЭ по русскому типа 2 из Решу ЕГЭ. Текст, выделенные слова, пять толкований, ответ — склейка номеров верных вариантов, например `35`. Нулевая база лежит в [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026).

Construct validity и anti-SLM objective нельзя смешивать без контроля. Совпадение показывают эмпирически на battery моделей при неизменном ключе $y^\star$.

Human pilot и цикл валидации людьми в проекте отсутствуют. Приёмка item опирается на печатный filter и на battery при temperature 0.

## 2. Датасет

Поля строки: `id`, `passage` с выделениями `**слово**`, `highlighted_words`, `options`, `answer`, `answer_list`, `explanation`. Инструкция задания не дублируется в строках. Она живёт на карточке датасета и в коде как `FIXED_INSTRUCTION`.

Eval подставляет ту же формулировку в промпт, просит ответ только цифрами и сравнивает с ключом exact match.

## 3. Печатное ограничение

Правки допустимы лишь пока текст остаётся грамматически и пунктуационно корректным русским, годным для печати. Unicode-lookalike, невидимые символы, опечатки ради атаки и любая порча бланка запрещены.

Battery — фиксированный набор малых моделей, на которых одновременно проверяют трудность. Декодирование только temperature 0.

## 4. Атака и TextAttack

Item $x$ задаёт контекст $c$, опции $O$ и ключ $y^\star$. Модель при temp 0 выдаёт строку ответа $\hat{y}$. Цель hardening — сдвинуть распределение ответов от $y^\star$ к неверным цифровым строкам, не ломая печатную норму и не меняя ключ.

Пусть $\mathcal{C}(x)$ — окрестность печатно корректных правок $x$, а $B$ — battery. Ищем

$$x^{\mathrm{adv}} \in \arg\max_{x' \in \mathcal{C}(x)} \sum_{m \in B} \left( \max_{a \neq y^\star} s_m(a \mid x') - s_m(y^\star \mid x') \right),$$

где $s_m$ — score ответа после промпта, на практике заменяемый индикатором ошибки $\mathbf{1}[\hat{y}_m(x') \neq y^\star]$ при greedy-декодировании. Так максимизируют зазор между неверными answer-строками и $y^\star$ под print constraint.

Схема совпадает с TextAttack: goal — ошибка модели на $y^\star$, transformation — разрешённые грамматические правки контекста и distractors, constraint — печатная корректность русского, search — дискретный локальный поиск по шагам. Character-noise и Unicode-атаки из классических recipe не берутся.

## 5. Вердикт

Отдельные мотивы известны. Selection bias MCQ, annotation artifacts, дискретная adversarial-оптимизация текста, AFLite-фильтры лёгких примеров уже описаны. Новизна не в них по отдельности.

Рабочий зазор — пайплайн под русский конструкт «значение слова в контексте» с прогоном на tip 2, без human pilot, с жёстким печатным ограничением и с battery при temperature 0. Вклад возможен как композиция и адаптация.

## 6. Литература

Zheng et al., ICLR 2024 — selection bias и PriDe. Pezeshkpour и Hruschka — разрыв accuracy при перестановке опций. Gururangan et al. — annotation artifacts. McCoy et al. — HANS. Sakaguchi et al. и Le Bras et al. — AFLite. Ebrahimi et al. — HotFlip. Morris et al. — TextAttack. GradQuiz и смежные работы — adversarial distractors в образовании. Jain и Wallace, Wiegreffe и Pinter — осторожность с attention как объяснением. Полные ссылки в конце файла.

Прямой перенос опечаток и Unicode-атак разрушает экзаменационный конструкт. Их не используем.

## 7. Вне скоупа

Сейчас в пайплайне выпуска нет human validity panel, двойной экспертной разметки, IRT-гейта по человеческой трудности $b_j$, пилота на школьниках для human accuracy $H$ и DIF по демографиям. Эти блоки требуют людей и могут вернуться позже.

## 8. Методы

Пайплайн: кандидаты, печатный grammatical filter, adversarial hardening на battery при temp 0, surface-filter, zero-edit и hardened eval. Ключ $y^\star$ при hardening не меняют, если иное не оговорено явно.

### M1. Sense-frequency traps

Берут слово с несколькими senses. Контекст однозначен для целевого sense $s^\star$, а частотный sense остаётся лексически соблазнительным для LM. В опциях — верное толкование, толкование частотного sense и словарьно допустимые ближние distractors. Метрика — падение exact-match на battery при том же ключе.

### M2. Grammatical paraphrase контекста

Парафраз $c \to c'$ сохраняет смысл ключа и пунктуационную норму. Меняют порядок слов, синонимы в пределах нормы, синтаксис. Так сбивают collocational cues и lexical overlap. Кандидаты фильтруют чекером нормы или сильной LM-as-judge только на грамматику и сохранение ключа. Принимают вариант, если battery accuracy падает.

### M3. Option-order как probe

Перестановки опций не публикуют как новый ключ бланка. Это зонд устойчивости ошибки при $\pi(O)$. Item считают model-hard по смыслу, только если ошибка держится на нескольких перестановках.

$$\mathrm{Consist}(x) = \frac{1}{R} \sum_{r=1}^{R} \mathbf{1}\left[\hat{y}\left(\pi_r(O)\right) = y^\star\right].$$

Низкая Consist к ключу при temp 0 на battery указывает на смысловую неустойчивость модели, а не только на order bias.

### M4. Distractor rewriting

Фиксируют $c$ и $y^\star$. Переписывают неверные опции так, чтобы они оставались словарно и грамматически валидными, но сильнее тянули LM prior к частотному или соседнему sense. Бессмысленные строки, опечатки и Unicode-подмены запрещены.

$$\max_{d} \sum_{m \in B} \mathbf{1}\left[\hat{y}_m(c, O[d]) \neq y^\star\right]$$

при печатной корректности $d$. Transfer проверяют на held-out модели вне battery.

### M5. Discrete near-paraphrase search

Локальный поиск по окрестности item. Один шаг — одна разрешённая правка. После правки текст обязан оставаться грамматически и пунктуационно корректным. Цель — минимизировать accuracy battery при неизменном $y^\star$. Это TextAttack-поиск с constraint «printable exam Russian». Остановка по бюджету шагов или по плато метрики.

### M6. AFLite-like filter

Строят слабые пути: только опции, только целевые слова, bag-of-words overlap с верной опцией. Items, которые weak path или слабая модель стабильно решает при temp 0 без полного контекста, помечают как surface-solvable и убирают из hard slice. Основной фильтр — сама battery на урезанных входах, а не линейный классификатор на эмбеддингах. Human agreement в приёмку не входит.

### Композиция

M1 задаёт пул с контролируемым sense. M2 и M5 правят контекст. M4 правит distractors. M3 проверяет, что трудность не сводится к порядку опций. M6 снимает surface-solvable хвост. Eval всегда включает zero-edit прогон всего набора при temperature 0.

## 9. Метрики

Exact-match accuracy $M$ — доля items, где нормализованный ответ модели совпал с ключом. Ответ сводят к множеству цифр 1–5 в каноническом порядке.

95% CI считают Wilson score interval и bootstrap по items.

Shuffle Consist — диагностика order bias, не замена exact-match.

Options-only и context-ablated accuracy ловят утечку смысла в опции или игнор контекста.

ASR — доля items, где hardening заставило модель ошибиться относительно zero-edit базы. Без печатного фильтра ASR не интерпретируют.

Weak-solvable rate — доля items, решаемых урезанным входом. После M6 показатель должен падать.

Transfer — те же items на модели вне battery.

Human accuracy и $\kappa$ agreement в текущем плане не собирают. Claim ограничивают устойчивостью к SLM при печатной норме, а не психометрической валидностью для людей.

## 10. Протокол eval

Модели по умолчанию — Qwen2.5 малого размера. Held-out — одна модель другой семьи близкого размера.

Декодирование только temperature 0, `do_sample=False`. Один фиксированный промпт с требованием цифр в формате ключа. Без few-shot с утечкой того же слова.

Базы прогона:

1. Zero-edit на всём [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026).
2. После hardening с тем же ключом и печатно корректными правками.
3. Order probe перестановками опций.
4. Ablation options-only и context-ablated.

MVP без людей:

1. Зафиксировать zero-edit baseline на полном наборе tip 2.
2. На items с высоким $M$ применить M2, M4, M5 под печатным constraint.
3. Прогнать M3 и M6.
4. Сравнить $M$, CI, Consist, weak-solvable rate, transfer.

Успех MVP: на hard slice средний $M$ battery ниже zero-edit базы, Consist к ключу не объясняется одной перестановкой, текст остаётся грамматически нормальным, ключ неизменен.

## 11. Риски

Без human pilot нельзя утверждать школьную валидность. Claim ограничивают anti-SLM устойчивостью при печатной норме.

Слишком редкий sense или узкий контекст могут ударить по разным группам учеников сильнее, чем по модели. Автоматический фильтр частоты sense снижает риск и не заменяет DIF.

Генераторы невидимых Unicode-perturbations и ключи закрытых КИМ не публикуют.

Hard set нестабилен без периодического refresh battery.

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
