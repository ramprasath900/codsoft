import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
df = pd.read_csv(r"D:\spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.3,
    stratify=df['label'],
    random_state=42
)

tfidf = TfidfVectorizer(stop_words='english')
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)
model = MultinomialNB()
model.fit(X_train_t, y_train)
y_pred = model.predict(X_test_t)
print("=== Naive Bayes + TF-IDF Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
