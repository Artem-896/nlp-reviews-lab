import json
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# 1. Підготовка ресурсів NLTK та spacy
nltk.download('punkt')
nltk.download('stopwords')

try:
  nlp = spacy.load('uk_core_news_sm')
except Exception:
  print(
      'Завантажте українську модель: python -m spacy download uk_core_news_sm'
  )

# --- ЗАВДАННЯ 1: Симуляція експорту відгуків (наприклад, з Rozetka) ---
reviews = [
    {
        'id': 1,
        'text': (
            'Дуже крутий телефон! Камера знімає чудово, батарея тримає весь'
            ' день. Рекомендую до покупки!'
        ),
    },
    {
        'id': 2,
        'text': (
            'Товар прийшов пошкодженим, упаковка була розірвана. Жахливий'
            ' сервіс доставки, більше не замовлятиму.'
        ),
    },
    {
        'id': 3,
        'text': (
            'Нормальний товар за свої гроші. Працює стабільно, але пластик'
            ' корпусу міг би бути кращим.'
        ),
    },
]

stop_words = {
    'і',
    'в',
    'на',
    'з',
    'до',
    'не',
    'що',
    'це',
    'як',
    'а',
    'але',
    'за',
    'вони',
    'він',
    'вона',
    'ми',
    'ви',
    'я',
    'ти',
}
stemmer = SnowballStemmer('russian')  # Для кирилиці

processed_data = []

print('--- Обробка відгуків (Токенізація, Стоп-слова, Стеммінг, Лематизація) ---')

for item in reviews:
  text = item['text']

  # А. Токенізація
  tokens = word_tokenize(text.lower())

  # Б. Видалення стоп-слів та розділових знаків
  filtered = [w for w in tokens if w.isalnum() and w not in stop_words]

  # В. Стеммінг
  stemmed = [stemmer.stem(w) for w in filtered]

  # Г. Лематизація (початкова форма через spacy)
  doc = nlp(text)
  lemmas = [
      token.lemma_ for token in doc if not token.is_stop and not token.is_punct
  ]

  processed_data.append({
      'id': item['id'],
      'original_text': text,
      'tokens': filtered,
      'stemmed': stemmed,
      'lemmatized': lemmas,
  })

# Збереження у JSON та CSV
with open('processed_reviews.json', 'w', encoding='utf-8') as f:
  json.dump(processed_data, f, ensure_ascii=False, indent=4)

df = pd.DataFrame(processed_data)
df.to_csv('processed_reviews.csv', index=False, encoding='utf-8-sig')

print('✓ Файли processed_reviews.json та processed_reviews.csv збережено!\n')

# --- ЗАВДАННЯ 2: Визначення тональності через TextBlob ---
print('--- Аналіз тональності відгуків (TextBlob) ---')

sentiment_counts = {'позитивний': 0, 'нейтральний': 0, 'негативний': 0}

for item in reviews:
  text = item['text']

  # Оскільки TextBlob точніший для англійської мови, перекладаємо перед оцінкою
  try:
    blob_en = TextBlob(text).translate(from_lang='uk', to='en')
    polarity = blob_en.sentiment.polarity
  except Exception:
    # Якщо перекладач недоступний, вимірюємо напростець
    polarity = TextBlob(text).sentiment.polarity

  if polarity > 0.1:
    tag = 'позитивний'
  elif polarity < -0.1:
    tag = 'негативний'
  else:
    tag = 'нейтральний'

  sentiment_counts[tag] += 1
  print(f"Відгук №{item['id']}: Оцінка = {polarity:.2f} -> {tag.upper()}")

print('\nПідсумкова статистика:')
print(f"Позитивних: {sentiment_counts['позитивний']}")
print(f"Нейтральних: {sentiment_counts['нейтральний']}")
print(f"Негативних: {sentiment_counts['негативний']}")