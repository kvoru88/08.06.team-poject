#!/usr/bin/env python3
"""
run.py — Точка входа для запуска MediaPulse приложения

Использование:
    python run.py              # Запуск на http://localhost:5000
    python run.py --port 8000  # Запуск на другом порту
"""

import argparse
import logging
from app import app
from database import init_db
from scheduler import init_scheduler_with_app

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Главная функция для запуска приложения"""
    
    parser = argparse.ArgumentParser(
        description="MediaPulse — Интеллектуальный агрегатор новостей"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5000,
        help="Порт сервера (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Хост сервера (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Запустить в режиме отладки"
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Запустить без APScheduler"
    )
    
    args = parser.parse_args()
    
    # Баннер
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 10 + "🚀 MediaPulse — Агрегатор новостей" + " " * 13 + "║")
    logger.info("║" + " " * 15 + "v1.0.0 | Flask + ML" + " " * 23 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    
    # Инициализация БД
    logger.info("📝 Инициализация базы данных...")
    init_db()
    logger.info("✓ БД инициализирована")
    
    # Инициализация scheduler'a
    if not args.no_scheduler:
        logger.info("⏰ Инициализация APScheduler...")
        scheduler = init_scheduler_with_app(app)
    else:
        scheduler = None
        logger.warning("⚠️  Scheduler отключён")
    
    # Информация о запуске
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info(f"║ 🌐 Веб-интерфейс: http://{args.host}:{args.port:<5} " + " " * 30 + "║")
    logger.info("║ 📚 API: http://localhost:5000/api/articles              ║")
    logger.info("║ 📊 Статистика: http://localhost:5000/api/stats         ║")
    logger.info("║ 🔗 Категории: http://localhost:5000/api/categories    ║")
    logger.info("║                                                        ║")
    logger.info(f"║ 🐛 Debug режим: {'ON' if args.debug else 'OFF':<44} ║")
    logger.info(f"║ ⏰ Scheduler: {'ON' if not args.no_scheduler else 'OFF':<43} ║")
    logger.info("║                                                        ║")
    logger.info("║ Нажмите Ctrl+C для остановки                          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    
    try:
        # Запуск Flask
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=False  # Отключаем для работы с scheduler
        )
    except KeyboardInterrupt:
        logger.info("\n")
        logger.info("🛑 Остановка приложения...")
        
        if scheduler:
            logger.info("🔴 Остановка scheduler'a...")
            scheduler.shutdown()
            logger.info("✓ Scheduler остановлен")
        
        logger.info("")
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║" + " " * 15 + "✓ Приложение остановлено" + " " * 19 + "║")
        logger.info("║" + " " * 10 + "Спасибо за использование MediaPulse!" + " " * 11 + "║")
        logger.info("╚" + "═" * 58 + "╝")
    except Exception as e:
        logger.error(f"Ошибка при запуске: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())