# Задания на значение слова в контексте против малых нейросетевых решателей

## 1. Зачем и постановка задачи

Project unsolvable исследует фреймворк генерации тестовых MCQ на понимание значения слова в тексте так, чтобы человек опирался на смысл, а малая языковая модель ошибалась из-за поверхностных эвристик. Рабочий прогон и валидация идут на реальных заданиях ЕГЭ по русскому того же типа: контекст, целевое слово, выбор верного толкования среди фиксированных опций.

Нельзя смешивать две цели. **Construct validity** — соответствие теста заявленному навыку. Целевой экзаменационный конструкт здесь это лексико-семантическая компетенция человека. Anti-NN objective, то есть цель заставить нейросетевой решатель ошибиться, это другой конструкт. Их совпадение нужно доказывать эмпирически, а не предполагать. Item с огромным успехом против модели и неадекватной трудностью для людей не годится как экзаменационный материал.

Ниже задача разобрана последовательно: формат и то, как модель его читает, вердикт по научной обоснованности идеи, уже известные методы литературы, открытые слоты пайплайна, угрозы модели, методы генерации и hardening, метрики, экспериментальный протокол с MVP, риски и этика, затем BibTeX. Термин вводится коротко при первом появлении и дальше используется без повтора определения.

## 2. Как модель видит задачу

Нас интересует **MCQ**, то есть multiple-choice question: задание с одним верным ответом среди фиксированного списка вариантов. Одна готовая единица такого теста называется **item**. В нашем типе item состоит из **контекста**, короткого фрагмента текста, **целевого слова** внутри него и нескольких **опций**, то есть вариантов толкования с буквенными метками. Неверные, но правдоподобные опции называют **distractors**. Пример: в предложении «Он сдал ключ от квартиры соседу» целевое слово *ключ*, верная опция «предмет для отпирания замка», а distractor «источник воды» тянет к другому значению того же слова.

Значение многозначного слова называют **sense**. У *коса* есть sense «орудие» и sense «причёска». Задача выбрать верный sense из инвентаря смыслов называется **WSD**, word sense disambiguation. Близкая постановка **WiC**, Word-in-Context, даёт два контекста для одного слова и спрашивает, совпадает ли смысл. Бенчмарк WiC описали Pilehvar и Camacho-Collados, многоязычное расширение XL-WiC дали Raganato et al. Русского среза в открытых списках XL-WiC нет, поэтому наш формат ближе к экзаменационному MCQ с толкованиями, чем к готовому WiC-датасету. Прогон делаем на ЕГЭ.

Модели, с которыми мы работаем, это **LLM**, large language models, крупные языковые модели. Современные LLM построены на архитектуре **transformer**, где фрагменты входа взаимодействуют через механизм внимания и слои MLP. Текст перед подачей в модель режется на **tokens**, атомарные куски: целые слова, части слов или знаки. Нас особенно интересуют **AR-модели**, autoregressive language models: они предсказывают следующий token по уже прочитанному префиксу. Малая AR-модель семейства Qwen2.5 решает MCQ, оценивая вероятность метки ответа после промпта с заданием. Сырой балл по метке до превращения в вероятность называется **logit**. Чем выше logit метки «B», тем охотнее модель выберет опцию B при прочих равных.

Пусть item задан как тройка контекст $c$, целевое слово $w$ и набор опций $O$ из $K$ толкований, а $y^\star$ — индекс верного ответа. Малая AR-модель оценивает

$$
\hat{y} = \arg\max_k\, p_\theta\!\left(a_k \mid \mathrm{prompt}(c, w, O)\right)
$$

где $a_k$ — токен метки опции.

У модели есть **prior**, априорное предпочтение до полного учёта семантики задания. Систематический перекос предсказаний относительно целевого правила решения называют **bias**. Два важных вида bias для MCQ: **option-order bias**, когда ответ зависит от порядка опций в списке, и **selection bias**, когда модель предпочитает конкретные метки вроде A или C независимо от текста опций. Zheng et al. на ICLR 2024 показывают, что у многих LLM selection bias связан с token prior на эти метки. Pezeshkpour и Hruschka измеряют большие разрывы точности при перестановке опций. Li и Gao разбирают anchored bias к первой опции A у семейства GPT-2. Следствие прямое: item, который ломает модель только при фиксированном порядке A–D, измеряет артефакт селектора, а не трудность смысла.

Ещё один источник ложной лёгкости — **shortcut**, то есть shortcut learning: модель решает датасет по ложным корреляциям вместо целевого навыка. Типичный пример в нашем формате: выбрать толкование с наибольшим пересечением слов с контекстом и проигнорировать реальный sense. Такие следы часто возникают как **artifacts**, annotation artifacts, непреднамеренные статистические маркеры разметки. Gururangan et al. показали их в NLI, где baseline только по гипотезе часто работает без premises. McCoy, Pavlick и Linzen в наборе HANS строят контролируемые кейсы, где эвристики lexical overlap и родственные им дают неверный label, и модели уровня BERT на них падают.

Отдельный сильный prior для малых моделей на русском дают **collocation**, устойчивые соседства слов. Сочетание вроде *острый ключ* в музыкальном тексте тянет к нотному sense сильнее, чем к смыслу «от замка». Если контекст и distractors подобраны слабо, модель отвечает по collocation, а не по нужному sense. Низкая доля русского в pretraining усиливает collocational priors и surface heuristics у малых моделей.

Для человеческой стороны нужна **psychometrics**, методы измерения свойств заданий по ответам людей, и в частности **IRT**, item response theory. В модели Rasch вероятность верного ответа человека $i$ на задание $j$ равна

$$
P\!\left(X_{ij} = 1 \mid \theta_i, b_j\right) = \frac{1}{1 + \exp\!\left(-(\theta_i - b_j)\right)}
$$

где $\theta_i$ — способность, $b_j$ — трудность задания.

## 3. Вердикт по идее

Идея научно обоснована: позиционный bias при выборе опций MCQ, shortcut на артефактах разметки, дискретная adversarial-оптимизация текста и неверность attention как объяснения — это устоявшиеся эмпирические факты в NLP, а не гипотезы. Новизна не в самих мотивах. Ближайшие решения уже строят adversarial distractors для MCQ, фильтруют лёгкие примеры через AFLite-подобные процедуры и генерируют AI-hard задания в образовании.

Реальный исследовательский зазор — композиционный пайплайн под русскоязычный конструкт «значение слова в контексте» с оценкой на ЕГЭ по русскому, с жёсткой валидацией construct validity для людей и с батареей малых AR-моделей класса Qwen2.5 0.6B–3.5B, где измеряются одновременно accuracy, согласованность при перемешивании опций и чувствительность к occlusion, а не только падение accuracy у целевой модели.

Переоткрывать уже нечего: позиционный bias MCQ измерен и частично объяснён token prior, shortcut и hypothesis-only — стандарт диагностики, TextAttack-подобные атаки — стандарт анализа robustness, LOO и attention disagreement не новы как инструменты, adversarial distractors для anti-LLM MCQ есть в GradQuiz, AFLite-фильтр лёгких примеров существует, AI-hard генерация заданий есть в math и programming.

Вклад возможен только если явно сформулировать четыре акцента. Первый: адаптация к русскому конструкту «значение слова в контексте» и прогон на ЕГЭ с явной психометрикой и human gold, а не только model-fool rate. Второй: композиция generation, затем hardening против battery малых AR-моделей, затем human validity filter, затем метрики consistency под shuffle и occlusion, а не один adversary. Третий: таксономия failure modes именно для word-meaning-in-context MCQ у малых моделей, с измеряемыми probes на каждый режим. Четвёртый: разделение двух целей — бенчмарк robustness малых LLM и инструмент высокой оценки для людей. Без этих акцентов проект выглядит как рекомбинация известных блоков без нового claim.

## 4. Что уже есть в литературе

### 4.1. Position и order bias в MCQ у LLM

Zheng et al. на ICLR 2024 показывают, что современные LLM неустойчивы к перестановке опций из-за selection bias: априорная масса вероятности на токены идентификаторов опций A, B, C, D доминирует над позицией в строке. Они предлагают PriDe — оценка prior на малой пермутированной подвыборке и дебиасинг на оставшихся примерах без меток. Pezeshkpour и Hruschka в Findings of NAACL 2024 измеряют разрыв accuracy при перестановке опций порядка 13–85% на разных бенчмарках и связывают чувствительность с неопределённостью между top-2 или top-3 вариантами. Li и Gao в Findings of ACL 2025 дают механистический разбор anchored bias у семейства GPT-2, где предпочтение первой опции A локализуется в MLP value-векторах и attention heads. Приёмы лечат bias селектора. Они не генерируют семантически трудные русские items.

### 4.2. Shortcut, артефакты данных, adversarial filtering

Gururangan et al. в NAACL 2018 показывают annotation artifacts в NLI: гипотеза-only baseline часто решает задачу без premises за счёт лексических маркеров. McCoy, Pavlick и Linzen в ACL 2019 вводят HANS: контролируемые кейсы, где эвристики lexical overlap, subsequence и constituent дают неверный label, и модели уровня BERT падают почти до нуля на таких кейсах. Sakaguchi et al. в AAAI 2020 для WinoGrande предлагают **AFLite** — лёгкий adversarial filter по эмбеддингам, удаляющий примеры, предсказуемые слабыми линейными классификаторами. Le Bras et al. в ICML 2020 формализуют AFLite как процедуру снижения измеримых dataset biases и показывают падение accuracy моделей при сохранении высокой человеческой точности, например с 92% до 62% на SNLI. Phang et al. в DADC 2022 предупреждают: adversarial evaluation sets сложнее, но ранжирование моделей нестабильно и зависит от adversary, а AFLite перепредставляет примеры с низким согласием аннотаторов. Фильтр «модель ошиблась» без human agreement даёт шум, а не валидный ЕГЭ-item.

### 4.3. Adversarial NLP и дискретная оптимизация

**Adversarial example** — вход, намеренно изменённый так, чтобы модель ошиблась при сохранении условия валидности для человека или формальных ограничений. Поиск таких правок в тексте это **discrete optimization**: замены символов или слов, а не непрерывный сдвиг пикселей. Ebrahimi et al. в ACL 2018 вводят HotFlip: градиентный flip символа или слова по one-hot входу с beam search. Li et al. в EMNLP 2020 предлагают BERT-Attack: ранжирование уязвимых слов по влиянию на logit целевой модели и замена через masked LM. Morris et al. в EMNLP 2020 представляют TextAttack как модульный каркас goal, constraints, transformation, search. Классическая цель этих методов — атаковать чтение текста классификатором. В образовательной постановке цель обратная: получить item, на котором модель ошибается, а человек нет. Прямой перенос опечаток и Unicode-подмен разрушает экзаменационный конструкт.

### 4.4. Anti-LLM задания в образовании

GradQuiz Bonifazi et al. в ACM TIST использует градиенты целевой LLM для возмущения семантически влиятельных сущностей и построения adversarial distractors для MCQ, снижая accuracy моделей на MMLU и OpenTriviaQA при сохранении педагогической связности по human evaluation. Xie et al. в Findings of EMNLP 2024 генерируют adversarial math word problems через структурные правки чисел по AST. Salim et al. в EMNLP 2024 мешают LLM-assisted cheating на intro programming через adversarial perturbation формулировок. Журналистские прогоны ChatGPT по ЕГЭ по русскому фиксируют ошибки на части форматов, но это не peer-reviewed бенчмарк. CHECK-MAT Khrulev оценивает VLM на проверке рукописных решений ЕГЭ по математике, то есть grading, а не генерацию resistant items. Ближайшие системы не заточены под русский конструкт «значение слова в контексте» с прогоном на ЕГЭ и не связывают hardening с IRT и human gold одновременно.

### 4.5. Attention faithfulness, leave-one-out, occlusion

**Leave-one-out**, сокращённо LOO, оценивает важность элемента через удаление одного token или фрагмента и измерение изменения предсказания. **Occlusion** маскирует часть входа с той же целью. Jain и Wallace в NAACL 2019 показывают слабую корреляцию attention с gradient и LOO, а также существование альтернативных attention-распределений с тем же предсказанием. Wiegreffe и Pinter в EMNLP-IJCNLP 2019 отвечают, что вывод зависит от определения explanation. Neely et al. показывают, что разные XAI-методы часто не согласуются между собой, поэтому agreement не является метрикой faithfulness. Madsen et al. в Findings of EMNLP 2022 оценивают faithfulness через Recursive ROAR и находят зависимость от модели и задачи. Occlusion и LOO полезны как behavioral probes чувствительности к токенам контекста, но не как доказательство «внимание смотрит не туда» внутри transformer. Для фреймворка LOO уместен как сигнал переписки контекста и как метрика: если удаление дистракторных collocates сильно меняет выбор модели, а удаление критического контекста для смысла слова — слабо, item эксплуатирует surface cues.

### 4.6. Психометрическая генерация и русский якорь

Tomikawa и коллеги связывают генерацию MCQ с целевым $b$ через LLM и DPO. DiVERT и overgenerate-and-rank для math MCQ строят distractors из представлений ошибок учащихся. Для целевого конструкта нужны distractors как типичные человеческие ошибки значения в контексте, а не только как градиентные adversarial foils. Иначе падает construct validity: item измеряет устойчивость к LLM, а не лексико-семантическую компетенцию. Научный якорь типа задания — WiC или WSD. Форматный якорь — MCQ с опциями значений.

### 4.7. Итог по уже существующему

Позиционный bias MCQ, shortcut, AFLite, TextAttack-подобные атаки, adversarial distractors вроде GradQuiz и WiC-рамка смысла в контексте уже известны. Отдельного академического пайплайна «значение слова в контексте на русском, anti-NN MCQ с human psychometrics и прогоном на ЕГЭ» в найденной литературе нет. Вклад возможен как композиция и адаптация, а не как переоткрытие отдельных блоков.

## 5. Что ещё можно сделать

Открытое пространство — собрать генерацию, hardening, человеческий фильтр и оценку на наборе моделей под русский word-meaning-in-context MCQ. **Hardening** здесь означает доработку уже сгенерированного item так, чтобы модели ошибались чаще без разрушения валидности для людей. **Model battery** — фиксированный набор решателей, на которых одновременно проверяют трудность, чтобы не подогнать задание под одну модель. **Human validity filter** — этап, где эксперты или пилот на людях принимают только задания с верным ключом, допустимыми distractors и приемлемой человеческой точностью.

### Слот A. Sense-controlled generation с контролем collocation

Генерация item, где целевой sense $s^\star$ однозначно читается из контекста, а более частотный sense остаётся лексически соблазнительным для AR-модели. WiC проверяет совпадение смыслов между двумя контекстами. Здесь нужен экзаменационный MCQ с толкованиями и явной ловушкой на collocational prior малой русской LM. Малые AR-модели сильно тянутся к частотным совместным появлениям, поэтому это главный содержательный удар по Qwen2.5-классу.

### Слот B. Battery-aware adversarial distractors без orthographic noise

Дискретный поиск толкований-дистракторов, максимизирующий ошибки сразу на наборе моделей при жёстком human accept. GradQuiz и HotFlip дают технику. Нужна адаптация к пространству русских толкований, запрет опечаток и Unicode, плюс оптимизация под battery, а не под одного adversary. Один foil может обмануть одну модель и не перенестись на held-out семью.

### Слот C. Order-robust и surface-robust приёмка item

Кандидат считается model-hard только если ошибка сохраняется при перестановках опций и после отсечения weak-solvable сигналов в духе AFLite. PriDe и AFLite уже есть как инструменты. Слот задаёт правило приёмки в генераторе: не принимать ложную трудность от метки A или от lexical overlap. Иначе датасет измеряет selection bias, а не смысл слова.

### Слот D. LOO-guided переписка контекста под human-critical span

Эксперт задаёт sense-critical span $S^\star$. Система измеряет важность фрагмента $t$ как разницу вероятности верного ответа с полным контекстом $c$ и с контекстом без $t$:

$$
I_m(t) = \left| p_m\!\left(y^\star \mid c\right) - p_m\!\left(y^\star \mid c \setminus t\right) \right|
$$

и переписывает контекст, если модель чувствительна к collocates вне $S^\star$, а человек опирается на $S^\star$. LOO как анализ известен. Здесь он становится редакционным циклом генерации с человеческим якорем. Так бьют context underuse без бессмысленных искажений текста.

### Слот E. IRT-гейт поверх anti-LLM objective

Даже при высоком успехе против модели item отбрасывается, если человеческая трудность $b_j$ вне целевого коридора или дискриминация низка. Контролируемая генерация вопросов по IRT существует. Связка «сначала anti-LLM hardening, потом психометрический гейт для русского конструкта с прогоном на ЕГЭ» в собранном виде отсутствует. Этот слот разводит бенчмарк robustness моделей и валидный измеритель школьной лексико-семантической компетенции.

Композиция: слот A даёт пул, B и D делают hardening, C чистит order и surface артефакты, E вместе с human validity filter закрывает выпуск.

## 6. Угрозы модели

Для малых Qwen2.5-класса особенно сильны collocational prior, ID bias и context underuse. Reasoning-модели частично бьют elimination shortcuts, но это уже другой adversary.

| Режим отказа | Механизм | Симптом |
|---|---|---|
| Lexical overlap | Перекрытие токенов между контекстом и верной опцией | Accuracy падает при paraphrase опции без смены смысла |
| Positional или ID prior | Bias на метку опции или на позицию в списке | Accuracy нестабильна при перестановке опций |
| Collocational prior | Целевое слово и соседи активируют частый sense из LM | Верный редкий sense проигрывает частотному |
| Surface heuristics | Длина опции, отрицания, канцеляризмы, формулировки «все кроме» | Hypothesis-only или options-only baseline выше chance |
| Option-elimination shortcuts | Исключение явно абсурдных foils | Двухшаговый prompt резко поднимает accuracy |
| Distractor attraction | Adversarial или collocational foil ближе к LM prior | Модель стабильно выбирает один и тот же foil |
| Context underuse | Решение по целевому слову без контекста | Mask или shuffle контекста почти не меняет ответ |
| Spurious cue words | Служебные маркеры в контексте коррелируют с ключом в датасете | AFLite-like classifier предсказывает ключ без полного контекста |

## 7. Методы пайплайна

Пайплайн: генерация кандидатов, adversarial hardening, human validity filter, model battery eval.

### M1. Sense-controlled generation с контролем collocates

Из русского словаря или WordNet-подобных ресурсов выбирают слово $w$ с не менее чем двумя senses. Контекст пишется так, чтобы единственный допустимый sense был целевой $s^\star$, а более частотный sense оставался лексически правдоподобным. Опции: толкование $s^\star$, толкование частотного sense, два ближних distractors по embedding sense. Модель максимизирует совместную вероятность слова и соседей из pretraining и тянется к частотному sense. Слишком редкий sense или искусственный контекст снижают fair difficulty и fairness по регионам и учебным программам. Измерение: human accuracy $H$, model accuracy $M$, gap $\Delta = H - M$. Collocation probe: замена соседей целевого слова на нейтральные при сохранении $s^\star$ должна поднимать $M$, если модель сидела на collocates.

### M2. Option-order и label-robust item acceptance

Кандидат принимается только если батарея моделей ошибается на не менее чем $m$ из $R$ случайных перестановок $O$, либо если после PriDe-подобной коррекции prior ошибка сохраняется. Формально

$$
\mathrm{Consist}(x) = \frac{1}{R} \sum_r \mathbf{1}\!\left[\hat{y}\!\left(\pi_r(O)\right) = y^\star\right]
$$

Item model-hard требует низкой Consist относительно верного ответа, то есть модель не находит $y^\star$ стабильно. Приёмка отсекает ложную трудность от ID bias. Риск в том, что авторы могут чинить порядок опций под экзамен и создать fairness-дыру между бланками. Измерение: RStd recall по позициям как у Zheng et al., плюс доля items, где $M$ падает только при фиксированном порядке.

### M3. AFLite-подобная фильтрация surface-solvable items

Строят weak learners на урезанных представлениях: bag-of-words $c$, только $w$, только опции, lexical overlap features. Итеративно удаляют примеры с высокой предсказуемостью weak score. Оставляют hard slice, где weak accuracy близка к chance, а human остаётся высокой. Убирает items, решаемые без контекстуализации смысла. Как у Phang et al.: риск перепредставленности спорных примеров, падение agreement, искажение рангов моделей. Измерение: weak-solvable rate, human agreement $\kappa$, корреляция difficulty с agreement. Не принимать item при $\kappa$ ниже порога.

### M4. Adversarial distractor search под battery моделей

Фиксируют $c$, $w$, $y^\star$. Ищут foils $d$ дискретным поиском: синонимические и sense-близкие замены в пространстве толкований, градиентные подсказки по малой white-box модели как в GradQuiz или HotFlip-стиле по токенам опций, black-box запросы к battery. Цель

$$
\max_d \sum_m \mathbf{1}\!\left[\hat{y}_m = d\right] \quad \text{при} \quad \mathrm{HumanAccept}(d) = 1
$$

Ограничения: грамматичность, семантическая близость к правдоподобной ошибке ученика, запрет на опечатки и Unicode-трюки. Создаёт аттракторы в пространстве опций, куда стекает LM prior. Слишком привлекательный foil повышает человеческую ошибку и ломает IRT-дискриминацию. Unicode и character noise недопустимы для экзаменационного конструкта. Измерение: attack success rate ASR на battery, human error rate на тех же foils, transfer ASR на held-out модели.

### M5. Occlusion- и LOO-guided hardening контекста

Для модели $m$ считают важность токена $t$ в $c$ формулой $I_m(t)$ из слота D. Если максимум $I_m$ лежит на нерелевантных collocates, а не на sense-critical span $S^\star$, переписывают $c$: усиливают $S^\star$, ослабляют spurious collocates, сохраняя $y^\star$ для людей. Альтернатива — генерировать $c$ так, чтобы у ошибающейся модели $I_m$ на $S^\star$ был низким, а у человека решение требовало $S^\star$. Бьёт context underuse и collocational prior напрямую. Переоптимизация под LOO конкретной модели даёт brittle items и неверный педагогический сигнал. Измерение: alignment score между $S^\star$ и top-$k$ по $I_m$. Доля items, где human accuracy падает при маскировании $S^\star$, а model accuracy почти нет.

### M6. IRT-aware контроль трудности для людей

Пилот на людях, оценка $b_j$ и дискриминации. Целевой коридор $b$ для экзаменационного уровня. Items вне коридора или с низкой дискриминацией отбрасывают, даже если ASR высок. Без этого фреймворк оптимизирует anti-LLM, а не измеряет русский лексический навык. Малый пилот даёт шумные $b_j$. Измерение: Rasch fit, item-total correlation, DIF по демографиям при наличии данных.

**Композиция.** M1 генерирует пул. M4 hardening distractors. M5 правит контекст. M3 и M2 фильтруют surface и order-артефакты. M6 и human panel — финальный validity gate. Затем battery eval. Коротко: generation задаёт конструкт, hardening бьёт малые AR-решатели, human filter сохраняет смысл для людей, battery eval доказывает, что трудность не сводится к order bias и annotation artifacts.

## 8. Метрики

**Accuracy модели $M$** — доля items, где модель выбрала верный ключ. Низкий $M$ на hard slice желателен только вместе с высоким человеческим результатом.

**Accuracy человека $H$** — доля items, где человек целевой группы выбрал верный ключ. Если $H$ падает вместе с $M$, hardening ударил по конструкту, а не только по модели.

**Gap** $\Delta = H - M$ растёт полезно только при стабильном $H$.

**Shuffle consistency** — доля перестановок опций, на которых модель всё же находит верный ответ, формула Consist выше. Для model-hard item низкая Consist к верному ключу означает, что модель не извлекает смысл стабильно. Если $M$ низок только при одном порядке и высок при других, это order artifact. **RStd**, стандартное отклонение recall по позициям опций, дополнительно сигналит selection или order bias.

**Options-only accuracy** — точность, когда контекст убран и остаются опции или целевое слово с опциями. Если она заметно выше случайного уровня, item содержит surface leakage.

**Context-ablated accuracy** — точность при маскировании или перемешивании контекста. Если абляция почти не меняет ответ, модель не использует контекст.

**Occlusion sensitivity к $S^\star$** — падение $H$ и $M$ при маскировании expert span, плюс корреляция $I_m$ с $S^\star$. Для валидного item человек должен зависеть от $S^\star$ сильнее, чем модель, сидящая на collocates.

**ASR**, attack success rate, — доля случаев, где атака или hardening заставляет модель ошибиться. ASR без $H$ и agreement читать нельзя.

**Weak-solvable rate** — доля items, которые weak learner решает по урезанным признакам. После AFLite-подобной чистки показатель должен падать.

**Human agreement $\kappa$** — согласованность экспертов по ключу и допустимости distractors. Порог задают заранее. Item с высоким ASR и низким $\kappa$ в выпуск не берут.

**Transfer ASR** — успех тех же items на held-out модели другой семьи. Без transfer результат может быть overfit к battery.

Опционально: PriDe-debiased accuracy, чтобы отделить semantic fail от ID bias.

Ориентиры минимального среза читать как рабочие гипотезы: на hard slice желательно $H \gtrsim 0.85$, средний $M$ battery ниже исходного пула, Consist к верному ответу низкая, transfer сохраняется, weak-solvable rate падает. Любой порог пересматривают после пилота на людях.

## 9. Экспериментальный протокол и MVP

### 9.1. Модели

Battery: Qwen2.5 порядка 0.5B–0.6B, 1.5B, 3B или ближайший доступный 3.5-класс. Отдельно instruction-tuned и base, если оба доступны. Held-out: одна модель другой семьи сопоставимого размера для transfer. Не смешивать с крупными reasoning-моделями в основном claim. Их можно держать как stress probe.

### 9.2. Данные

Серебряный старт: 200–400 MCQ word-in-context на русском, из открытых фрагментов и синтетики M1, с двойной экспертной разметкой смысла. Золотой пилот: не менее 50 items после полного пайплайна с human accuracy и agreement.

### 9.3. Базы сравнения

1. Исходные items ЕГЭ или шаблонные items того же формата без hardening.
2. Random distractors той же длины.
3. Embedding-nearest sense distractors без adversarial search.
4. GradQuiz-подобная генерация foils без русского sense-контроля.
5. Только AFLite-фильтр без adversarial foils.

### 9.4. Ablations и статистика

Убирать по одному: M2, M3, M4, M5, M6. Смотреть $\Delta$, Consist, human $\kappa$, transfer ASR. Главный критерий успеха MVP: высокий $H$, низкий $M$ на battery, стабильность при shuffle, transfer на held-out, $\kappa$ не хуже исходного пула. Bootstrap CI по items. Фиксация decoding: greedy или temperature 0. Несколько шаблонов prompt. Запрет few-shot с утечкой того же слова.

### 9.5. Минимальный MVP

Один тип задания: русское слово в одном предложении-контексте, 4 опции толкований, один верный смысл.

Срез работ:

1. Собрать 80–100 кандидатов вручную или полуавтоматом по M1 из частотных многозначных слов школьного круга.
2. Прогнать battery Qwen2.5-класса, измерить $M$, shuffle consistency, options-only и context-ablation baselines.
3. На 30–40 items с высоким $M$ применить M4 или M5, пересобрать foils или контекст.
4. Двое экспертов-филологов: верный ответ, допустимость foils, критический span $S^\star$. Порог согласия заранее.
5. Пилот на 15–30 людях целевой возрастной группы для грубой оценки $H$ и отсева токсичных foils.
6. Сравнить с базами из раздела 9.3. Зафиксировать, растёт ли $\Delta$ без падения $H$ и без роста shuffle-артефактов.

Остановка MVP успешна, если на hard slice $H \gtrsim 0.85$, средний $M$ battery заметно ниже исходного пула, Consist к верному ответу остаётся низкой, transfer на held-out сохраняется, а weak-solvable rate падает. Иначе вклад сводится к известному selection bias или к невалидным foils.

## 10. Риски и этика

**Fairness.** Adversarial фильтры и foils могут ударить по ученикам с иным словарным запасом, диалектным фоном или слабой подготовкой сильнее, чем по LLM. Нужен DIF-контроль и запрет orthographic noise.

**Construct validity.** Целевой конструкт — лексико-семантическая компетенция человека. Anti-LLM objective — другой конструкт. Их совпадение нужно доказывать эмпирически.

**Dual-use.** Тот же пайплайн усиливает exam cheating toolkits и adversarial атаки на классификаторы. Публиковать методы и метрики, не публиковать генератор невидимых Unicode-perturbations и готовые ключи закрытых КИМ. Не использовать закрытые официальные КИМ без разрешения правообладателя.

**Нестабильность бенчмарка.** Как у Phang et al., adversary-specific hard set устаревает при смене модели. Battery и periodic refresh обязательны.

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

@inproceedings{li2025anchored,
  title={Anchored Answers: Unravelling Positional Bias in {GPT}-2's Multiple-Choice Questions},
  author={Li, Ruizhe and Gao, Yanjun},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
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

@inproceedings{phang2022adversarialsets,
  title={Adversarially Constructed Evaluation Sets Are More Challenging, but May Not Be Fair},
  author={Phang, Jason and Chen, Angelica and Huang, William and Bowman, Samuel R.},
  booktitle={Proceedings of the First Workshop on Dynamic Adversarial Data Collection},
  year={2022}
}

@inproceedings{ebrahimi2018hotflip,
  title={{HotFlip}: White-Box Adversarial Examples for Text Classification},
  author={Ebrahimi, Javid and Rao, Anyi and Lowd, Daniel and Dou, Dejing},
  booktitle={Proceedings of ACL (Short Papers)},
  year={2018}
}

@inproceedings{li2020bertattack,
  title={{BERT-ATTACK}: Adversarial Attack Against {BERT} Using {BERT}},
  author={Li, Linyang and Ma, Ruotian and Guo, Qipeng and Xue, Xiangyang and Qiu, Xipeng},
  booktitle={Proceedings of EMNLP},
  year={2020}
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

@inproceedings{wiegreffe2019attention,
  title={Attention is not not Explanation},
  author={Wiegreffe, Sarah and Pinter, Yuval},
  booktitle={Proceedings of EMNLP-IJCNLP},
  year={2019}
}

@inproceedings{madsen2022roar,
  title={Evaluating the Faithfulness of Importance Measures in {NLP} by Recursively Masking Allegedly Important Tokens and Retraining},
  author={Madsen, Andreas and Meade, Nicholas and Adlakha, Vaibhav and Reddy, Siva},
  booktitle={Findings of EMNLP},
  year={2022}
}

@inproceedings{neely2021order,
  title={Order in the Court: Explainable {AI} Methods Prone to Disagreement},
  author={Neely, Michael and Schouten, Stefan F. and Bleeker, Maurits J. R. and Lucic, Ana},
  booktitle={ICML Workshop on Theoretic Foundation, Criticism, and Application Trend of Explainable AI},
  year={2021},
  note={arXiv:2105.03287}
}

@inproceedings{nie2020anli,
  title={Adversarial {NLI}: A New Benchmark for Natural Language Understanding},
  author={Nie, Yixin and Williams, Adina and Dinan, Emily and Bansal, Mohit and Weston, Jason and Kiela, Douwe},
  booktitle={Proceedings of ACL},
  year={2020}
}

@inproceedings{pilehvar2019wic,
  title={{WiC}: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations},
  author={Pilehvar, Mohammad Taher and Camacho-Collados, Jose},
  booktitle={Proceedings of NAACL-HLT},
  year={2019}
}

@inproceedings{raganato2020xlwic,
  title={{XL-WiC}: A Multilingual Benchmark for Evaluating Semantic Contextualization},
  author={Raganato, Alessandro and Pasini, Tommaso and Camacho-Collados, Jose and Pilehvar, Mohammad Taher},
  booktitle={Proceedings of EMNLP},
  year={2020}
}

@inproceedings{xie2024adversarialmath,
  title={Adversarial Math Word Problem Generation},
  author={Xie, Roy and Huang, Chengxuan and Wang, Junlin and Dhingra, Bhuwan},
  booktitle={Findings of EMNLP},
  year={2024}
}

@inproceedings{salim2024impeding,
  title={Impeding {LLM}-assisted Cheating in Introductory Programming Assignments via Adversarial Perturbation},
  author={Salim, Saiful Islam and Yang, Rubin Yuchan and Cooper, Alexander and Ray, Suryashree and Debray, Saumya and Rahaman, Sazzadur},
  booktitle={Proceedings of EMNLP},
  year={2024}
}

@article{bonifazi2026gradquiz,
  title={Leveraging Adversarial Attacks to Generate Multiple-Choice Quizzes Robust Against Large Language Models},
  author={Bonifazi, Gianluca and Buratti, Christopher and Marchetti, Michele and Parlapiano, Federica and Traini, Davide and Ursino, Domenico and Virgili, Luca},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2026},
  doi={10.1145/3803797}
}

@article{khrulev2025checkmat,
  title={{CHECK-MAT}: Checking Hand-Written Mathematical Answers for the Russian Unified State Exam},
  author={Khrulev, Ruslan},
  journal={arXiv preprint arXiv:2507.22958},
  year={2025}
}

@misc{unsolvable-guide-2026,
  title={Unified Guide for Neural-Solver-Resistant Word-in-Context {MCQ} with {EGE} Evaluation},
  author={{Project unsolvable}},
  howpublished={Project document: docs/unsolvable-guide.md},
  year={2026}
}
```

The project is under GPL-3.0 license.
