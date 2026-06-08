import sqlite3

DB_NAME = "news.db"

def init_db():
    """Создает базу данных и таблицу articles, если их еще нет"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text TEXT,
            url TEXT NOT NULL UNIQUE,  -- UNIQUE защищает от дубликатов новостей
            source TEXT,
            date TEXT,
            category TEXT DEFAULT 'Unknown' -- Заглушка, пока ML-щик не отдаст модель
        )
    ''')
    
    conn.commit()
    conn.close()
    print("База данных успешно инициализирована!")

def save_article(title, text, url, source, date, category='Unknown'):
    """Сохраняет одну новость. Если URL уже есть — просто пропускает"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO articles (title, text, url, source, date, category)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, text, url, source, date, category))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Ошибка при записи в БД: {e}")
    finally:
        conn.close()

def get_all_articles():
    """Запрос для Фуллстека: забрать все новости"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM articles ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()