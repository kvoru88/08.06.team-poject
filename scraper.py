import requests
from bs4 import BeautifulSoup
from database import save_article
from classifier import predict_category

# Настоящие заголовки, чтобы к коду не было вопросов
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'ru-RU,ru;q=0.9'
}

def scrape_section(url, source_name):
    """Универсальный и стабильный скрапер новостных лент"""
    print(f"Парсим ленту {source_name} через BeautifulSoup...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Находим новостные заголовки (универсальный поиск по тегам статей и ссылкам)
            articles = soup.find_all('a', href=True)
            
            count = 0
            for article in articles:
                title = article.text.strip()
                link = article['href']
                
                # Отсекаем слишком короткие ссылки/менюшки и берем только целевой контент
                if len(title) > 30 and not link.startswith('#'):
                    if not link.startswith('http'):
                        link = url + link
                    
                    # Передаем текст новости ИИ-классификатору твоего напарника
                    category = predict_category(title)
                    
                    # Сохраняем в твою базу данных SQLite
                    save_article(title, title, link, source_name, "Сегодня", category)
                    count += 1
                    
                # Нам достаточно 15-20 свежих новостей с каждого сайта для демонстрации
                if count >= 15:
                    break
                    
            print(f"✅ {source_name} успешно спарсен. Добавлено новостей: {count}")
        else:
            print(f"❌ Не удалось скачать {source_name}. Статус код: {response.status_code}")
    except Exception as e:
        print(f"💥 Ошибка при парсинге {source_name}: {e}")

def scrape_all_news():
    print("=== ЗАПУСК АВТОНОМНОГО HTML-СКРАПИНГА СИСТЕМЫ ===")
    
    # 3 стабильных источника (Технологии, Спорт, Политика/Общее) без жестких блокировок
    scrape_section("https://habr.com/ru/all/", "Хабр (Технологии)")
    scrape_section("https://news.rambler.ru/sport/", "Рамблер (Спорт)")
    scrape_section("https://news.rambler.ru/politics/", "Рамблер (Политика)")
    
    print("=== БЭКЕНД: СКРАПИНГ ПОЛНОСТЬЮ ЗАВЕРШЕН ===")

if __name__ == "__main__":
    scrape_all_news()