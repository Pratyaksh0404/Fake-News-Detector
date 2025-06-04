import sys
import joblib
from utils import preprocess_text


def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        model = joblib.load('models/fake_news_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        print("❌ Error: Model or vectorizer not found. Please train the model first.")
        sys.exit(1)


def predict_news(text):
    """
    Predict whether the given news article is fake or real.

    Args:
        text (str): Raw news article text

    Returns:
        str: "Fake" or "Real"
    """
    model, vectorizer = load_model_and_vectorizer()

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    print(text_vector.shape)
    print(text_vector)

    # Predict and probability
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]
    print("Raw text:", text)
    print("Processed:", processed_text)
    print("TF-IDF features:", text_vector.shape)
    print("Prediction:", pred)
    print("\n📊 Prediction Probabilities:")
    print(f"🟥 Fake: {probs[0]:.4f}")
    print(f"🟩 Real: {probs[1]:.4f}")

    result = "Fake" if pred == 0 else "Real"
    print(f"\n✅ Final Prediction: {result}")
    return result


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"Your news article text here\"")
        sys.exit(1)

    input_text = sys.argv[1]
    predict_news(input_text)


if __name__ == "__main__":
    main()
