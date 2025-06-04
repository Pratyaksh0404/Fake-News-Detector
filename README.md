# 📰 Fake News Detection Web App

Detect whether a news article is **REAL** or **FAKE** using powerful AI models — including both traditional ML and fine-tuned BERT. The app also performs source credibility checks, content analysis, and text summarization.

---

## 🚀 Project Highlights

### 🔍 Input Options:
- Enter raw text
- Submit a URL (article link)

### 🧠 Model Options:
- BERT (Transformers by Hugging Face)
- TF-IDF + Random Forest (Traditional ML)

### 📊 Output Features:
- Fake or Real prediction
- Confidence score visualized using a semi-circular gauge chart
- Article summary (if long)
- Source credibility analysis
- Content characteristics (e.g., satire, emotional tone, statistics, etc.)


---

## 🧰 Tech Stack
- Python (Flask, Scikit-learn, Transformers, NLTK)
- HTML/CSS/Bootstrap (Frontend UI)
- JavaScript + Chart.js (for gauge chart)
- Hugging Face Transformers (`bert-base-uncased`, `facebook/bart-large-cnn`)

---

## 📁 Folder Structure
- ├── app.py # Flask backend
- ├── templates/
- │ └── index.html # HTML frontend
- ├── models/
- │──└─ bert_fakenews1.pt # Trained BERT model (run notebook to generate)
- │──└─ fake_news_model.joblib # Trained Random Forest model (run script to generate)
- │──└─ tfidf_vectorizer.joblib
- ├── utils.py # Preprocessing, summarization, analysis functions
- ├── bert-model.ipynb # Notebook to train and save BERT model
- ├── model_train.py # Script to train TF-IDF + Random Forest model
- ├── requirements.txt

  
---

## 🛠️ Installation & Setup

1. Clone this repo:
    ```bash
    git clone https://github.com/yourusername/Fake-News-Detector.git
    cd Fake-News-Detector
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate    # Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download required NLTK resources:
    ```python
    # In Python shell
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5. Place model files:
    - Drop your `bert_fakenews1.pt`, `tfidf_vectorizer.joblib`, and `fake_news_model.joblib` inside the `models/` directory.

6. Run the app:
    ```bash
    python app.py
    ```
7. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## 📥 Download Training Dataset

Due to GitHub file size limits, please download the training data manually:

- [Fake.csv , True.csv (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

After downloading, place them in the `data/` folder before training.

---

## 📌 Features in Action

| Feature         | Description                                      |
|-----------------|------------------------------------------------|
| **BERT Model**  | Fine-tuned on custom fake/real news dataset    |
| **Random Forest** | Trained on TF-IDF vectors                      |
| **Summarizer**  | Uses BART model (`facebook/bart-large-cnn`)    |
| **Source Analysis** | Uses domain-based credibility lookup         |
| **Content Insights** | Detects satire, emotion, clickbait, and more |
| **Confidence Score** | Displayed via modern gauge chart              |

---

## 🤝 Contributing

This is a personal project, but feel free to fork and explore!

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and build upon it.

---

## 🙌 Acknowledgements

- Hugging Face Transformers  
- NLTK  
- Flask  
- Chart.js  
- Inspiration: Real-world media literacy tools & research on misinformation

---

## 📣 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/pratyaksh-agrawal-59b82928a/)  

