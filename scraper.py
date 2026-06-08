import feedparser
from database import save_article

# 3 источника по ТЗ (Технологии, Спорт, Политика/Общее)
SOURCES = {
    "Хабр (Технологии)": "https://habr.com/ru/rss/all/all/",
    "Спорт-Экспресс (Спорт)": "https://www.sport-express.ru/services/rss/news/se/",
    "РБК (Политика/Общее)": "https://rssexport.rbc.ru/rbc/news/3/full.news.rss"
}

def scrape_news():
    print("Робот пошел собирать новости...")
    
    for source_name, url in SOURCES.items():
        print(f"Сканируем: {source_name}")
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            title = entry.get('title', 'Без заголовка')
            link = entry.get('link', '')
            text = entry.get('summary', entry.get('description', 'Нет описания'))
            date = entry.get('published', 'Нет даты')
            
            # Пока ML-щик не сделал классификатор, ставим 'Unknown'
            category = "Unknown"
            
            # Сохраняем в твою базу данных
            save_article(title, text, link, source_name, date, category)
            
    print("Готово! Все свежие новости в базе.")

if __name__ == "__main__":
    scrape_news()