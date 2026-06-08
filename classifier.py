"""
classifier.py — ML-модуль классификации новостей для проекта MediaPulse.

Реализует полный pipeline машинного обучения:
    текст → очистка → TF-IDF → LogisticRegression → категория

Категории: спорт | политика | технологии

Использование:
    # Обучение модели (один раз):
    train_model("dataset.csv")

    # Предсказание в рантайме:
    category = predict_category("OpenAI представила новую языковую модель")

Автор: ML Engineer — MediaPulse Team
"""

import os
import re
import logging

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ─── Настройка логирования ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Пути к артефактам модели ─────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "news_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# ─── Стоп-слова русского языка (базовый набор) ───────────────────────────────
RUSSIAN_STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
    "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
    "у", "же", "вы", "за", "бы", "по", "только", "её", "мне", "было",
    "вот", "от", "меня", "ещё", "нет", "о", "из", "ему", "теперь",
    "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
    "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
    "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
    "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "при", "наконец", "два", "об",
    "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою",
    "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "новый", "новая", "новое", "новые", "своего", "стал", "стало",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ПРЕДОБРАБОТКА ТЕКСТА
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Очищает и нормализует текст новости для подачи в векторизатор.

    Этапы обработки:
        1. Приведение к нижнему регистру.
        2. Удаление URL-адресов.
        3. Удаление HTML-тегов.
        4. Удаление небуквенных символов (оставляем только кириллицу/латиницу).
        5. Удаление стоп-слов русского языка.
        6. Удаление одиночных букв.
        7. Нормализация пробелов.

    Args:
        text (str): Исходный текст статьи (заголовок + описание).

    Returns:
        str: Очищенный текст, пригодный для TF-IDF векторизации.

    Example:
        >>> preprocess("OpenAI представила новую языковую модель GPT-5!")
        'openai представила языковую модель gpt'
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Нижний регистр
    text = text.lower()

    # 2. Удаление URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Удаление HTML-тегов
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Оставляем только буквы (кириллица + латиница) и пробелы
    text = re.sub(r"[^а-яёa-z\s]", " ", text)

    # 5. Токенизация и фильтрация стоп-слов
    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in RUSSIAN_STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ОБУЧЕНИЕ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

def train_model(dataset_path: str = "dataset.csv") -> dict:
    """Обучает модель классификации новостей на размеченном датасете.

    Pipeline обучения:
        CSV → DataFrame → preprocess() → TF-IDF → LogisticRegression → .pkl

    TF-IDF (Term Frequency–Inverse Document Frequency):
        Преобразует текст в числовой вектор, где каждое измерение соответствует
        слову из словаря. Значение = TF(слово) × IDF(слово).
        - TF — частота слова в данном документе.
        - IDF — обратная частота слова во всём корпусе (редкие слова важнее).
        Это позволяет выделить ключевые термины каждой категории.

    LogisticRegression:
        Линейный классификатор, обучающий весовые векторы для каждого класса.
        Выбран потому что:
        - Высокая интерпретируемость весов признаков.
        - Устойчивость при высокой размерности (TF-IDF создаёт тысячи признаков).
        - Быстрое обучение и инференс.
        - Отличная baseline-точность на текстовых задачах (~85–95 %).

    Args:
        dataset_path (str): Путь к CSV-файлу с колонками ``text`` и ``category``.

    Returns:
        dict: Словарь с метриками::

            {
                "accuracy": 0.93,
                "report": "...",      # classification_report
                "classes": ["политика", "спорт", "технологии"],
            }

    Raises:
        FileNotFoundError: Если датасет не найден по указанному пути.
        ValueError: Если в датасете отсутствуют колонки ``text`` или ``category``.
    """
    logger.info("═" * 55)
    logger.info("  MediaPulse — Обучение ML-классификатора")
    logger.info("═" * 55)

    # ── Загрузка данных ───────────────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    df = pd.read_csv(dataset_path)
    logger.info("Загружено записей: %d", len(df))

    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV должен содержать колонки 'text' и 'category'.")

    # Удаляем пропуски
    df.dropna(subset=["text", "category"], inplace=True)
    df = df[df["text"].str.strip() != ""]
    logger.info("Записей после очистки: %d", len(df))

    # Распределение по классам
    logger.info("Распределение классов:\n%s", df["category"].value_counts().to_string())

    # ── Предобработка текста ─────────────────────────────────────────────────
    logger.info("Предобработка текстов...")
    df["clean_text"] = df["text"].apply(preprocess)

    # ── Train/Test split ──────────────────────────────────────────────────────
    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── TF-IDF Векторизатор ──────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Топ-5000 слов по TF-IDF весу
        ngram_range=(1, 2),      # Униграммы + биграммы ("новая модель")
        sublinear_tf=True,       # Логарифмическое сглаживание TF
        min_df=1,                # Минимальная частота слова в корпусе
        analyzer="word",
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info("Размер словаря TF-IDF: %d", len(vectorizer.vocabulary_))

    # ── Логистическая регрессия ───────────────────────────────────────────────
    model = LogisticRegression(
        C=1.0,          # Обратная сила регуляризации L2
        max_iter=1000,  # Максимум итераций оптимизатора
        solver="lbfgs", # Quasi-Newton метод — оптимален для малых датасетов
        random_state=42,
    )

    logger.info("Обучение LogisticRegression...")
    model.fit(X_train_vec, y_train)

    # ── Оценка модели ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=sorted(df["category"].unique()),
    )

    logger.info("─" * 45)
    logger.info("  ACCURACY: %.4f (%.1f %%)", acc, acc * 100)
    logger.info("─" * 45)
    logger.info("\nClassification Report:\n%s", report)

    # ── Сохранение артефактов ──────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info("Модель сохранена:      %s", MODEL_PATH)
    logger.info("Векторизатор сохранён: %s", VECTORIZER_PATH)

    return {
        "accuracy": round(acc, 4),
        "report": report,
        "classes": sorted(df["category"].unique()),
        "vocab_size": len(vectorizer.vocabulary_),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ЗАГРУЗКА МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Загружает обученную модель и векторизатор из файлов .pkl.

    Используется при каждом запуске Flask-приложения, чтобы не переобучать
    модель заново. Артефакты хранятся в директории ``models/``.

    Returns:
        tuple: Пара ``(model, vectorizer)``::

            model       — обученный LogisticRegression
            vectorizer  — обученный TfidfVectorizer

    Raises:
        FileNotFoundError: Если файлы .pkl не найдены.
            Решение: сначала вызвать ``train_model()``.

    Example:
        >>> model, vectorizer = load_model()
        >>> vec = vectorizer.transform(["текст новости"])
        >>> model.predict(vec)
        array(['технологии'], dtype=object)
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Модель не найдена. Запустите train_model() для обучения.\n"
            f"  Ожидаемые файлы:\n  {MODEL_PATH}\n  {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Модель загружена из %s", MODEL_PATH)
    return model, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# 4. ПРЕДСКАЗАНИЕ КАТЕГОРИИ
# ─────────────────────────────────────────────────────────────────────────────

def predict_category(text: str) -> str:
    """Предсказывает тематическую категорию для текста новости.

    Полный pipeline предсказания:
        1. ``preprocess(text)``         — очистка текста
        2. ``vectorizer.transform()``   — TF-IDF векторизация
        3. ``model.predict()``          — классификация
        4. Возврат строки-категории

    Args:
        text (str): Произвольный текст новости (заголовок, описание или оба).

    Returns:
        str: Категория из набора: ``"спорт"``, ``"политика"``, ``"технологии"``.
             Возвращает ``"неизвестно"`` если текст пуст или модель не загружена.

    Example:
        >>> predict_category("OpenAI представила новую языковую модель")
        'технологии'

        >>> predict_category("Сборная выиграла чемпионат мира по футболу")
        'спорт'

        >>> predict_category("Президент подписал новый закон")
        'политика'
    """
    clean = preprocess(text)
    if not clean:
        logger.warning("predict_category: пустой текст после предобработки.")
        return "неизвестно"

    try:
        model, vectorizer = load_model()
    except FileNotFoundError as exc:
        logger.error("predict_category: %s", exc)
        return "неизвестно"

    vector = vectorizer.transform([clean])
    category = model.predict(vector)[0]

    # Уверенность предсказания (вероятность выбранного класса)
    proba = model.predict_proba(vector).max()
    logger.info(
        "predict_category | вход: %.50s... | категория: %s | уверенность: %.2f%%",
        text, category, proba * 100,
    )
    return category


def predict_with_confidence(text: str) -> dict:
    """Предсказывает категорию и возвращает вероятности по всем классам.

    Полезно для Flask API или отладки, когда нужно знать распределение
    вероятностей по всем категориям, а не только победившую.

    Args:
        text (str): Текст новости.

    Returns:
        dict: Словарь с результатами::

            {
                "category":     "технологии",
                "confidence":   0.94,
                "probabilities": {
                    "политика":    0.03,
                    "спорт":       0.03,
                    "технологии":  0.94,
                },
            }

    Example:
        >>> result = predict_with_confidence("Apple выпустила новый iPhone")
        >>> result["category"]
        'технологии'
        >>> result["confidence"]
        0.97
    """
    clean = preprocess(text)
    if not clean:
        return {"category": "неизвестно", "confidence": 0.0, "probabilities": {}}

    model, vectorizer = load_model()
    vector = vectorizer.transform([clean])

    category = model.predict(vector)[0]
    proba_array = model.predict_proba(vector)[0]
    probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(model.classes_, proba_array)
    }

    return {
        "category": category,
        "confidence": round(float(proba_array.max()), 4),
        "probabilities": probabilities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ТОЧКА ВХОДА — ДЕМОНСТРАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Шаг 1: Обучение ───────────────────────────────────────────────────────
    dataset_file = "dataset.csv"
    if not os.path.exists(dataset_file):
        logger.error("Файл датасета '%s' не найден.", dataset_file)
        sys.exit(1)

    metrics = train_model(dataset_file)

    print("\n" + "═" * 55)
    print("  РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ")
    print("═" * 55)
    print(f"  Accuracy : {metrics['accuracy'] * 100:.1f} %")
    print(f"  Классы   : {', '.join(metrics['classes'])}")
    print(f"  Словарь  : {metrics['vocab_size']} токенов")
    print(f"  Train    : {metrics['train_size']} записей")
    print(f"  Test     : {metrics['test_size']} записей")
    print("═" * 55)

    # ── Шаг 2: Демонстрация predict_category ─────────────────────────────────
    test_news = [
        ("OpenAI представила новую языковую модель GPT-5",          "технологии"),
        ("Сборная России победила в финале чемпионата мира",        "спорт"),
        ("Президент подписал закон о цифровой экономике",           "политика"),
        ("Tesla запустила автопилот на нейронных сетях",            "технологии"),
        ("Боксёр выиграл титул чемпиона мира по версии WBC",        "спорт"),
        ("Государственная дума приняла новый федеральный бюджет",   "политика"),
        ("Nvidia анонсировала видеокарту для ИИ-вычислений",        "технологии"),
        ("Хоккейная сборная завоевала золото Олимпийских игр",      "спорт"),
    ]

    print("\n  ДЕМОНСТРАЦИЯ КЛАССИФИКАЦИИ НОВОСТЕЙ")
    print("─" * 55)
    print(f"  {'Новость':<44} {'Результат':<12} {'Верно?'}")
    print("─" * 55)

    correct = 0
    for news_text, expected in test_news:
        result = predict_with_confidence(news_text)
        predicted = result["category"]
        conf = result["confidence"]
        is_correct = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        short_text = (news_text[:41] + "...") if len(news_text) > 44 else news_text
        print(f"  {short_text:<44} {predicted:<12} {is_correct}  ({conf:.0%})")

    print("─" * 55)
    print(f"  Demo accuracy: {correct}/{len(test_news)} ({correct / len(test_news):.0%})")
    print("═" * 55)
    print("\n  Модель готова к интеграции с Flask и SQLite.")
    print("  Файлы артефактов:")
    print(f"    {MODEL_PATH}")
    print(f"    {VECTORIZER_PATH}")
    print("═" * 55)
