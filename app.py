"""
app.py — Flask приложение для агрегатора новостей MediaPulse

Реализует:
    ✓ Главная страница со всеми новостями
    ✓ Поиск по ключевому слову
    ✓ Фильтрация по категориям
    ✓ REST API для получения статей
    ✓ Dark Mode (Cyberpunk UI)
    ✓ Интеграция с ML-классификатором

Маршруты:
    /                    — главная страница
    /api/articles        — получить все статьи (с фильтрацией/поиском)
    /api/articles/<id>   — получить одну статью
    /api/categories      — список категорий
"""

import sqlite3
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from classifier import predict_category, predict_with_confidence

# ─── Настройка логирования ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Инициализация Flask ───────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False  # Поддержка кириллицы в JSON

DB_NAME = "news.db"

# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    """Возвращает подключение к БД"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def articles_from_db(
    query="SELECT * FROM articles ORDER BY id DESC",
    params=()
):
    """Получает список статей из БД и преобразует в список словарей"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    articles = [dict(row) for row in rows]
    return articles

def search_articles(keyword, category=None):
    """Поиск по ключевому слову и фильтрация по категории"""
    query = "SELECT * FROM articles WHERE (title LIKE ? OR text LIKE ?)"
    params = [f"%{keyword}%", f"%{keyword}%"]
    
    if category and category != "все":
        query += " AND category = ?"
        params.append(category)
    
    query += " ORDER BY id DESC"
    return articles_from_db(query, tuple(params))

def filter_by_category(category):
    """Фильтр по категории"""
    if category == "все":
        return articles_from_db()
    return articles_from_db(
        "SELECT * FROM articles WHERE category = ? ORDER BY id DESC",
        (category,)
    )

def get_categories():
    """Получить уникальные категории из БД"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM articles ORDER BY category")
    rows = cursor.fetchall()
    conn.close()
    
    categories = [row[0] for row in rows if row[0]]
    return sorted(categories)

def get_trending_topics(limit=10):
    """Получить тренды дня (топ статьи)"""
    return articles_from_db(
        "SELECT * FROM articles ORDER BY id DESC LIMIT ?",
        (limit,)
    )

# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЕ МАРШРУТЫ
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Главная страница с лентой новостей"""
    logger.info("GET / — Загрузка главной страницы")
    
    categories = get_categories()
    articles = articles_from_db()
    total_count = len(articles)
    
    return render_template(
        "index.html",
        articles=articles,
        categories=categories,
        total_count=total_count,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/article/<int:article_id>")
def article_detail(article_id):
    """Страница одной статьи"""
    logger.info(f"GET /article/{article_id} — Просмотр статьи")
    
    articles = articles_from_db(
        "SELECT * FROM articles WHERE id = ?",
        (article_id,)
    )
    
    if not articles:
        return render_template("404.html", message="Статья не найдена"), 404
    
    article = articles[0]
    return render_template("article.html", article=article)

# ─────────────────────────────────────────────────────────────────────────────
# REST API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/articles", methods=["GET"])
def api_get_articles():
    """
    API: Получить статьи с поиском и фильтрацией
    
    Query параметры:
        ?search=ключевое_слово  — поиск по заголовку и тексту
        ?category=спорт         — фильтр по категории
        ?limit=20               — максимум результатов
        ?offset=0               — пропустить N результатов (пагинация)
    
    Пример:
        /api/articles?search=AI&category=технологии&limit=10
    """
    search = request.args.get("search", "").strip()
    category = request.args.get("category", "все").strip()
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    logger.info(f"API: Получение статей | search={search} | category={category}")
    
    try:
        # Поиск и фильтр
        if search:
            articles = search_articles(search, category)
        elif category and category != "все":
            articles = filter_by_category(category)
        else:
            articles = articles_from_db()
        
        # Пагинация
        total = len(articles)
        articles = articles[offset : offset + limit]
        
        return jsonify({
            "success": True,
            "count": len(articles),
            "total": total,
            "offset": offset,
            "articles": articles
        })
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/articles/<int:article_id>", methods=["GET"])
def api_get_article(article_id):
    """API: Получить одну статью по ID"""
    try:
        articles = articles_from_db(
            "SELECT * FROM articles WHERE id = ?",
            (article_id,)
        )
        
        if not articles:
            return jsonify({
                "success": False,
                "error": "Статья не найдена"
            }), 404
        
        return jsonify({
            "success": True,
            "article": articles[0]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/categories", methods=["GET"])
def api_get_categories():
    """API: Получить все категории"""
    try:
        categories = get_categories()
        return jsonify({
            "success": True,
            "categories": ["все"] + categories
        })
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/trending", methods=["GET"])
def api_get_trending():
    """API: Получить топ статьи дня"""
    limit = request.args.get("limit", 5, type=int)
    try:
        articles = get_trending_topics(limit)
        return jsonify({
            "success": True,
            "trending": articles
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/stats", methods=["GET"])
def api_get_stats():
    """API: Получить статистику по новостям"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Общее количество
        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]
        
        # По категориям
        cursor.execute(
            "SELECT category, COUNT(*) as count FROM articles GROUP BY category"
        )
        by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        # По источникам
        cursor.execute(
            "SELECT source, COUNT(*) as count FROM articles GROUP BY source"
        )
        by_source = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return jsonify({
            "success": True,
            "total": total,
            "by_category": by_category,
            "by_source": by_source
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN ENDPOINTS (для интеграции с scheduler)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/admin/reclassify", methods=["POST"])
def api_reclassify():
    """
    Переклассифицировать все статьи с категорией 'Unknown' или 'неизвестно'
    Используется scheduler'ом после скрапинга новых статей
    """
    logger.info("API: Начало переклассификации статей")
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Получить все статьи с Unknown
        cursor.execute(
            "SELECT id, title, text FROM articles WHERE category IN ('Unknown', 'неизвестно')"
        )
        articles_to_classify = cursor.fetchall()
        
        classified_count = 0
        for article_id, title, text in articles_to_classify:
            # Объединяем заголовок и текст для лучшей классификации
            combined_text = f"{title} {text}"
            category = predict_category(combined_text)
            
            cursor.execute(
                "UPDATE articles SET category = ? WHERE id = ?",
                (category, article_id)
            )
            classified_count += 1
            logger.info(f"Переклассифицирована статья {article_id}: {category}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Переклассифицировано {classified_count} статей")
        return jsonify({
            "success": True,
            "message": f"Переклассифицировано {classified_count} статей",
            "count": classified_count
        })
    except Exception as e:
        logger.error(f"Error in reclassify: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/admin/scrape", methods=["POST"])
def api_scrape():
    """
    Запустить скрапинг вручную (используется scheduler'ом)
    """
    logger.info("API: Начало скрапинга")
    
    try:
        from scraper import scrape_news
        scrape_news()
        
        return jsonify({
            "success": True,
            "message": "Скрапинг завершён успешно"
        })
    except Exception as e:
        logger.error(f"Error in scrape: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    """404 ошибка"""
    return render_template("404.html", message="Страница не найдена"), 404

@app.errorhandler(500)
def server_error(error):
    """500 ошибка"""
    logger.error(f"Server Error: {error}")
    return render_template("500.html", message="Ошибка сервера"), 500

# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК ПРИЛОЖЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  MediaPulse — Запуск Flask приложения")
    logger.info("=" * 60)
    
    # Инициализация БД
    from database import init_db
    init_db()
    
    # Инициализация APScheduler
    from scheduler import init_scheduler_with_app
    scheduler = init_scheduler_with_app(app)
    
    # Запуск сервера
    try:
        app.run(
            debug=False,
            host="127.0.0.1",
            port=5000,
            use_reloader=False  # Отключаем для работы с scheduler
        )
    except KeyboardInterrupt:
        logger.info("\nОстановка приложения...")
        scheduler.shutdown()
        logger.info("✓ Scheduler остановлен")
    finally:
        logger.info("Приложение завершено")