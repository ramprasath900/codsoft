import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GenreClassifier:
    def __init__(self, path):
        self.df = self._load_and_clean(path)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000)

    def _load_and_clean(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(" ::: ")
                if len(parts) == 4:
                    _, _, genre, plot = parts
                    data.append((plot.strip(), genre.strip().lower()))
        df = pd.DataFrame(data, columns=["plot", "genre"])
        valid_genres = df['genre'].value_counts()[lambda x: x >= 100].index
        return df[df['genre'].isin(valid_genres)]

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df['plot'], self.df['genre'], test_size=0.2, stratify=self.df['genre'])

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        y_pred = self.model.predict(X_test_vec)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {round(acc * 100, 2)}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=sorted(self.model.classes_))
        self._save_confusion_matrix(cm)

        # Show top 5 confusions
        self._print_top_confusions(cm, sorted(self.model.classes_))

        # Save predictions
        result_df = pd.DataFrame({
            'plot': X_test,
            'actual': y_test.values,
            'predicted': y_pred
        })
        result_df.to_csv("predicted_results.csv", index=False)
        print("📄 Predictions saved to: predicted_results.csv")

    def _save_confusion_matrix(self, cm):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=sorted(self.model.classes_),
                    yticklabels=sorted(self.model.classes_))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("🖼️ Confusion matrix saved as: confusion_matrix.png")

    def _print_top_confusions(self, cm, labels):
        print("\n🔍 Top 5 Most Confused Genre Pairs:")
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)  # ignore correct predictions
        flat = [(labels[i], labels[j], cm_copy[i][j])
                for i in range(len(labels)) for j in range(len(labels)) if cm_copy[i][j] > 0]
        top_confusions = sorted(flat, key=lambda x: -x[2])[:5]
        for a, b, count in top_confusions:
            print(f"'{a}' predicted as '{b}' → {count} times")

# Run
if __name__ == "__main__":
    GenreClassifier("D:/train_data.txt").run()
