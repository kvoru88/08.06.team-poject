"""
scheduler.py — APScheduler для автоматизации обновления новостей

Реализует:
    ✓ Автоматический скрапинг каждый час
    ✓ Автоматическая классификация новых статей
    ✓ Логирование процесса
    ✓ Обработка ошибок

Интеграция с Flask:
    Запускается фоновым потоком вместе с Flask приложением
    Использует APScheduler для упланирования задач
"""

import logging
import sqlite3
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ─── Настройка логирования ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_NAME = "news.db"

# ─────────────────────────────────────────────────────────────────────────────
# ЗАДАЧИ SCHEDULER'A
# ─────────────────────────────────────────────────────────────────────────────

def scrape_and_classify_job():
    """
    Основная задача: скрапинг → классификация → сохранение
    
    Выполняет:
        1. Скрапинг новостей из источников
        2. Классификацию новых статей через ML
        3. Обновление категорий в БД
        4. Логирование результатов
    """
    logger.info("=" * 70)
    logger.info("  SCHEDULER: Начало цикла обновления новостей")
    logger.info(f"  Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    try:
        # Шаг 1: Скрапинг
        logger.info("SCHEDULER: Шаг 1 — Скрапинг новостей...")
        from scraper import scrape_news
        scrape_news()
        logger.info("✓ Скрапинг завершён")
        
        # Шаг 2: Классификация новых статей
        logger.info("SCHEDULER: Шаг 2 — Классификация новых статей...")
        classify_unknown_articles()
        logger.info("✓ Классификация завершена")
        
        # Шаг 3: Логирование статистики
        stats = get_database_stats()
        logger.info("SCHEDULER: Статистика базы данных:")
        logger.info(f"  📰 Всего статей: {stats['total']}")
        logger.info(f"  📊 По категориям: {stats['by_category']}")
        logger.info(f"  🔗 По источникам: {stats['by_source']}")
        
        logger.info("=" * 70)
        logger.info("  SCHEDULER: Цикл завершён успешно ✓")
        logger.info("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"✗ SCHEDULER ERROR: {e}", exc_info=True)
        logger.error("=" * 70 + "\n")

def classify_unknown_articles():
    """
    Переклассифицировать все статьи с категорией 'Unknown'
    используя обученный ML-классификатор
    """
    from classifier import predict_category
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Получить статьи без классификации
    cursor.execute(
        "SELECT id, title, text FROM articles WHERE category IN ('Unknown', 'неизвестно')"
    )
    articles = cursor.fetchall()
    
    if not articles:
        logger.info("  → Новых статей для классификации не найдено")
        return
    
    logger.info(f"  → Найдено {len(articles)} статей для классификации")
    
    classified_count = 0
    for article_id, title, text in articles:
        try:
            # Объединяем заголовок и текст для лучшей классификации
            combined_text = f"{title} {text if text else ''}"
            category = predict_category(combined_text)
            
            cursor.execute(
                "UPDATE articles SET category = ? WHERE id = ?",
                (category, article_id)
            )
            classified_count += 1
            
            logger.debug(f"    Статья #{article_id}: {category}")
            
        except Exception as e:
            logger.warning(f"    ✗ Ошибка при классификации #{article_id}: {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"  → Переклассифицировано: {classified_count} статей")

def get_database_stats():
    """Получить статистику по новостям в БД"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Всего статей
    cursor.execute("SELECT COUNT(*) FROM articles")
    total = cursor.fetchone()[0]
    
    # По категориям
    cursor.execute(
        "SELECT category, COUNT(*) FROM articles GROUP BY category"
    )
    by_category = {row[0]: row[1] for row in cursor.fetchall()}
    
    # По источникам
    cursor.execute(
        "SELECT source, COUNT(*) FROM articles GROUP BY source"
    )
    by_source = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total": total,
        "by_category": by_category,
        "by_source": by_source
    }

# ─────────────────────────────────────────────────────────────────────────────
# ИНИЦИАЛИЗАЦИЯ И ЗАПУСК SCHEDULER'A
# ─────────────────────────────────────────────────────────────────────────────

def init_scheduler():
    """
    Инициализирует и запускает APScheduler
    
    Расписание:
        - Каждый час: scrape_and_classify_job()
        - При запуске: сразу же запустить один раз
    """
    scheduler = BackgroundScheduler()
    
    # Добавляем основную задачу (каждый час)
    scheduler.add_job(
        func=scrape_and_classify_job,
        trigger=IntervalTrigger(hours=1),
        id="scrape_and_classify",
        name="Scrape and Classify News",
        replace_existing=True,
        max_instances=1,  # Только один экземпляр одновременно
    )
    
    # ОПЦИОНАЛЬНО: Запустить скрапинг сразу при старте
    # Раскомментируйте если хотите немедленное обновление при запуске
    # logger.info("SCHEDULER: Запуск первого цикла при старте...")
    # scheduler.add_job(
    #     func=scrape_and_classify_job,
    #     trigger="date",
    #     run_date=datetime.now(),
    #     id="first_run",
    # )
    
    # Запуск scheduler'а
    scheduler.start()
    logger.info("✓ APScheduler инициализирован и запущен")
    logger.info(f"  Следующий скрапинг: через ~1 час")
    
    return scheduler

# ─────────────────────────────────────────────────────────────────────────────
# ИНТЕГРАЦИЯ С FLASK
# ─────────────────────────────────────────────────────────────────────────────

def init_scheduler_with_app(app):
    """
    Инициализирует scheduler с Flask приложением
    
    Используется как:
        from scheduler import init_scheduler_with_app
        init_scheduler_with_app(app)
    """
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        func=scrape_and_classify_job,
        trigger=IntervalTrigger(hours=1),
        id="scrape_and_classify",
        name="Scrape and Classify News",
        replace_existing=True,
        max_instances=1,
    )
    
    scheduler.start()
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  APScheduler инициализирован с Flask приложением" + " " * 22 + "║")
    logger.info("║  Автоматическое обновление новостей: КАЖДЫЙ ЧАС" + " " * 20 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    return scheduler

# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК (ТЕСТИРОВАНИЕ)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("  TEST MODE: Запуск scheduler'a для проверки")
    logger.info("=" * 70 + "\n")
    
    # Запуск scheduler'a
    scheduler = init_scheduler()
    
    try:
        # Scheduler работает в фоне, обработчик ошибок в консоли
        import time
        logger.info("Scheduler работает (нажмите Ctrl+C для выхода)...\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n\nОстановка scheduler'a...")
        scheduler.shutdown()
        logger.info("✓ Scheduler остановлен")