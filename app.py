from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import numpy as np
from utils import preprocess_text, summarize_text, extract_text_from_url, analyze_text_characteristics, analyze_source_credibility
import os
from urllib.parse import urlparse

app = Flask(__name__)

# Load models and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
bert_model.load_state_dict(torch.load("models/bert_fakenews1.pt", map_location=torch.device("cpu")))
bert_model.eval()

rf_model = joblib.load("models/fake_news_model.joblib")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Updated credible domains
CREDIBLE_DOMAINS = [
    "thehindu.com", "timesofindia.indiatimes.com", "hindustantimes.com",
    "indiatoday.in", "bbc.com", "cnn.com", "reuters.com", "ndtv.com",
    "economictimes.indiatimes.com", "theguardian.com", "nytimes.com",
    "forbes.com", "bloomberg.com"
]

def enhanced_source_credibility_check(url):
    domain = urlparse(url).netloc.lower()
    for source in CREDIBLE_DOMAINS:
        if source in domain:
            return f"Source appears credible ({domain})"
    return f"Source credibility unknown ({domain})"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_type = request.form.get("input_type")
        model_choice = request.form.get("model_choice")
        user_input = request.form.get("user_input")

        # Extract content
        if input_type == "url":
            article_text = extract_text_from_url(user_input)
            if not article_text or len(article_text.strip().split()) < 30:
                return jsonify({"error": "Could not extract meaningful content from the URL."})
        else:
            article_text = user_input

        if not article_text or len(article_text.strip()) < 10:
            return jsonify({"error": "Couldn't extract sufficient content."})

        # Preprocess
        cleaned_text = preprocess_text(article_text)

        # Predict
        if model_choice == "bert":
            inputs = bert_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy().flatten()
            pred_label = int(np.argmax(probs))
            confidence = round(min(probs[pred_label] * 100, 96), 2)
        else:
            tfidf_features = tfidf_vectorizer.transform([cleaned_text])
            pred_label = int(rf_model.predict(tfidf_features)[0])
            confidence = round(min(np.max(rf_model.predict_proba(tfidf_features)) * 100, 99.5), 2)

        label = "REAL" if pred_label == 1 else "FAKE"

        # Summary
        summary = summarize_text(article_text) if len(article_text.split()) > 40 else None

        # Characteristics
        heuristics = analyze_text_characteristics(article_text)
        source_analysis = enhanced_source_credibility_check(user_input) if input_type == "url" else "N/A"

        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "summary": summary,
            "characteristics": heuristics,
            "source": source_analysis,
            "original": article_text
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
