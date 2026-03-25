import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from deep_translator import GoogleTranslator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib

matplotlib.use('TkAgg')

nltk.download('vader_lexicon', quiet=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def get_news_texts(limit=100):
    texts = []
    url = "https://www.pravda.com.ua/news/"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        links = [a['href'] for a in soup.select('.article_news_list .article_title a')][:limit]

        for link in links:
            if not link.startswith('http'):
                link = "https://www.pravda.com.ua" + link
            article_res = requests.get(link, headers=HEADERS, timeout=10)
            article_soup = BeautifulSoup(article_res.text, 'html.parser')
            container = article_soup.select_one('.post_news_text')

            if container:
                raw_text = " ".join([p.text.strip() for p in container.find_all('p')])
                clean_text = re.sub(r'\s+', ' ', re.sub(r'[^а-яіїєґА-ЯІЇЄҐa-zA-Z\s]', ' ', raw_text)).strip()
                if len(clean_text) > 100:
                    texts.append(clean_text.lower())
    except Exception as e:
        print(f"Помилка парсингу: {e}")
    return texts


def process_text_and_sentiment(texts):
    print("Завантаження моделей та аналіз тональності.")
    nlp = spacy.load("uk_core_news_sm")
    sia = SentimentIntensityAnalyzer()
    translator = GoogleTranslator(source='uk', target='en')

    results = []
    for text in texts:
        doc = nlp(text)
        cleaned_words = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        cleaned_text_for_cloud = " ".join(cleaned_words)

        # VADER аналіз тональності через машинний переклад
        try:
            # Перекладаємо перші 3000 символів, щоб не перевантажити API перекладача
            translated_text = translator.translate(text[:3000])

            # VADER повертає словник: 'neg', 'neu', 'pos', 'compound'
            sentiment_scores = sia.polarity_scores(translated_text)

            # Беремо пропорцію загальної нейтральності тексту
            neu_ratio = sentiment_scores['neu']
            compound_score = sentiment_scores['compound']

            # Якщо новина на 80% і більше складається з сухих фактів вона нейтральна
            if neu_ratio >= 0.80:
                category = 'Нейтральна'
            elif compound_score > 0:
                category = 'Позитивна'
            else:
                category = 'Негативна'
        except Exception as e:
            print(f"Помилка аналізу тексту: {e}")
            category = 'Нейтральна'

        results.append({
            'Джерело': 'pravda.com.ua',
            'Оригінальний_Текст': text,
            'Очищений_Текст': cleaned_text_for_cloud,
            'Тональність': category
        })

    return pd.DataFrame(results)


def olap_and_visualize(df):
    # Побудова OLAP-куба
    olap_cube = pd.pivot_table(df, values='Оригінальний_Текст', index=['Джерело', 'Тональність'], aggfunc='count').rename(columns={'Оригінальний_Текст': 'Кількість'})
    print("\nOLAP ЗВІТ:")
    print(olap_cube)

    plt.figure(figsize=(14, 6))

    # Діаграма тональності
    plt.subplot(1, 2, 1)
    sentiment_counts = df['Тональність'].value_counts()
    colors = {'Негативна': '#ff9999', 'Нейтральна': '#c2c2f0', 'Позитивна': '#99ff99'}
    pie_colors = [colors.get(x, '#333333') for x in sentiment_counts.index]

    if not sentiment_counts.empty:
        sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=pie_colors, startangle=90)
    plt.title('OLAP: Розподіл тональності новин', fontsize=14)
    plt.ylabel('')

    # Хмара слів
    plt.subplot(1, 2, 2)
    all_words = ' '.join(df['Очищений_Текст'].tolist())
    if all_words.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
    plt.title('Хмара слів:', fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Починаємо збір даних.")
    news_texts = get_news_texts()
    if news_texts:
        df_results = process_text_and_sentiment(news_texts)
        olap_and_visualize(df_results)