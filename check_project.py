#!/usr/bin/env python3
"""
Скрипт для проверки целостности проекта MediaPulse

Использование:
    python check_project.py
"""

import os
import sys

def check_files():
    """Проверяет наличие всех необходимых файлов"""
    
    required_files = {
        # Python файлы
        "app.py": "Flask приложение",
        "scheduler.py": "APScheduler интеграция",
        "scraper.py": "Скрапинг новостей",
        "database.py": "Работа с SQLite",
        "classifier.py": "ML классификатор",
        "run.py": "Точка входа",
        
        # Конфигурация
        "requirements.txt": "Зависимости проекта",
        "README.md": "Основная документация",
        "FULLSTACK_GUIDE.md": "Руководство для Fullstack разработчика",
        
        # Шаблоны
        "templates/base.html": "Базовый шаблон",
        "templates/index.html": "Главная страница",
        "templates/article.html": "Страница статьи",
        "templates/404.html": "Ошибка 404",
        "templates/500.html": "Ошибка 500",
        
        # Данные и модели
        "dataset.csv": "Датасет для обучения",
        "models/news_model.pkl": "Обученная модель",
        "models/vectorizer.pkl": "TF-IDF векторизатор",
    }
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "📋 Проверка целостности проекта" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    missing = []
    found = []
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            found.append((filepath, description))
            status = "✅"
        else:
            missing.append((filepath, description))
            status = "❌"
        
        print(f"{status} {filepath:<30} — {description}")
    
    print("\n" + "─" * 60)
    print(f"✅ Найдено: {len(found)}/{len(required_files)}")
    
    if missing:
        print(f"❌ Не найдено: {len(missing)}")
        print("\nОтсутствующие файлы:")
        for filepath, description in missing:
            print(f"  - {filepath}")
        return False
    
    return True

def check_python_version():
    """Проверяет версию Python"""
    print("\n" + "─" * 60)
    print("Python версия: " + sys.version.split()[0])
    
    if sys.version_info >= (3, 8):
        print("✅ Версия совместима (требуется 3.8+)")
        return True
    else:
        print("❌ Версия не совместима (требуется 3.8+)")
        return False

def check_dependencies():
    """Проверяет установленные зависимости"""
    print("\n" + "─" * 60)
    print("Проверка зависимостей...\n")
    
    required = [
        "flask",
        "feedparser",
        "beautifulsoup4",
        "apscheduler",
        "sklearn",
        "pandas",
        "joblib",
    ]
    
    installed = []
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
            installed.append(package)
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    print(f"\n✅ Установлено: {len(installed)}/{len(required)}")
    
    if missing:
        print(f"❌ Отсутствует: {len(missing)}")
        print("\nУстановите зависимости:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    print()
    
    # 1. Проверка файлов
    files_ok = check_files()
    
    # 2. Проверка Python версии
    python_ok = check_python_version()
    
    # 3. Проверка зависимостей
    deps_ok = check_dependencies()
    
    # Итог
    print("\n" + "╔" + "═" * 58 + "╗")
    
    if files_ok and python_ok and deps_ok:
        print("║" + " " * 20 + "✅ ВСЁ ГОТОВО!" + " " * 25 + "║")
        print("║" + " " * 10 + "Запустите: python run.py" + " " * 24 + "║")
    else:
        print("║" + " " * 15 + "⚠️  ТРЕБУЕТСЯ ПОДГОТОВКА" + " " * 19 + "║")
        if not files_ok:
            print("║  - Проверьте наличие всех файлов" + " " * 23 + "║")
        if not python_ok:
            print("║  - Обновите Python до версии 3.8+" + " " * 21 + "║")
        if not deps_ok:
            print("║  - Установите зависимости (pip install -r requirements.txt)" + " " * 4 + "║")
    
    print("╚" + "═" * 58 + "╝\n")
    
    return 0 if (files_ok and python_ok and deps_ok) else 1

if __name__ == "__main__":
    sys.exit(main())