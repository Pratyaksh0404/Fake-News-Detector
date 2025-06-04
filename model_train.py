# model_train.py
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import download_nltk_data, load_and_preprocess_data
from sklearn.model_selection import cross_val_score

def create_model_directory():
    if not os.path.exists("models"):
        os.makedirs("models")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()
    plt.close()

def main():
    print("📥 Checking NLTK dependencies...")
    download_nltk_data()

    print("📦 Loading and preprocessing data...")
    X, y = load_and_preprocess_data("data/Fake.csv", "data/True.csv")

    print("🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

    print("🧠 Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("🚀 Training model with RandomForest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    print("🧪 Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    print("📈 Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))
    print(f"Training Accuracy: {train_acc:.4f}")

    scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    print("CV Accuracy Scores:", scores)
    print("Mean Accuracy:", scores.mean())

    print("💾 Saving model and vectorizer...")
    create_model_directory()
    joblib.dump(model, "models/fake_news_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

    print("✅ Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
