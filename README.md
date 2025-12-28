#  Sentiment Analysis using Naive Bayes & TF-IDF

This project implements a **text sentiment classification system** using classical Natural Language Processing (NLP) techniques.  
The goal is to classify text reviews as **positive** or **negative** using a **TF-IDF vectorizer** and a **Multinomial Naive Bayes** classifier.

---

##  Project Overview

Sentiment analysis is a common NLP task used in applications such as:

- Product review analysis  
- Social media monitoring  
- Customer feedback systems  

This project demonstrates how **traditional machine learning techniques** can be effectively used to build a sentiment classifier **without deep learning**, while still following a clean and standard ML workflow.

---

##  Dataset Overview

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)  
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Total samples:** 50,000 movie reviews  
- **Positive reviews:** 25,000  
- **Negative reviews:** 25,000  
- Balanced binary sentiment classification dataset

---

##  Model Pipeline

The workflow follows standard machine learning best practices:

### 1. Text Preprocessing
- Lowercasing
- Stopword removal
- Tokenization (handled internally by TF-IDF)

### 2. Feature Extraction
- TF-IDF Vectorization
- Unigrams + Bigrams (`ngram_range=(1,2)`)

### 3. Model Training
- Multinomial Naive Bayes

### 4. Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### 5. Prediction
- Sentiment prediction on unseen text

---

## Technologies Used

- Python  
- Scikit-learn  
- NumPy  
- Jupyter Notebook  

---
=

## ⚙️ Model Configuration

![Model Configuration](screenshots/model_config.png)





- **max_features = 5000**
  - Limits vocabulary size to the most informative terms
  - Reduces noise and helps prevent overfitting

- **ngram_range = (1, 2)**
  - Uses unigrams and bigrams
  - Captures short contextual phrases like *“not good”* and *“very bad”*

- **stop_words = 'english'**
  - Removes common non-informative words
  - Improves signal-to-noise ratio in features

- **Overall Impact**
  - Provides a strong, efficient feature representation
  - Well-suited for classical ML models like Naive Bayes


---

## 📈 Model Evaluation

![Model Evaluation](screenshots/Evaluation.png)



- **Training Accuracy:** 86.53%  
- **Testing Accuracy:** 85.03%  
  → Indicates good generalization with minimal overfitting.

- **Class-wise Performance:**
  - Negative sentiment: Precision 0.86, Recall 0.84, F1-score 0.85
  - Positive sentiment: Precision 0.84, Recall 0.86, F1-score 0.85
  → Balanced performance across both classes.

- **Overall Metrics:**
  - Macro Avg F1-score: 0.85
  - Weighted Avg F1-score: 0.85
  → Confirms consistent performance despite class distribution.

- **Confusion Matrix Analysis:**
  - Correct predictions dominate both classes.
  - Misclassifications are symmetric, showing no strong class bias.

- **Key Takeaway:**
  - The model provides a strong baseline for sentiment analysis using TF-IDF + Naive Bayes, with stable and interpretable results.


---

##  Key Learnings

- Classical ML models depend heavily on **feature representation**
- TF-IDF captures **frequency**, not semantic meaning
- Naive Bayes performs well but has **limited contextual understanding**
- Evaluation metrics must always be interpreted with **model limitations** in mind

---

##  Future Improvements

To improve performance and realism, the following upgrades can be applied:

### 1. Advanced Models
- Logistic Regression  
- Linear SVM  
- XGBoost  

### 2. Better Text Representations
- Word2Vec  
- GloVe  
- FastText  

### 3. Deep Learning Approaches
- LSTM / GRU  
- Transformer-based models (BERT, RoBERTa, DistilBERT)

### 4. Dataset Enhancements
- Larger datasets  
- Improved class balance  
- Domain-specific text cleaning  

---
## Conclusion

This project demonstrates how classical NLP techniques like **TF-IDF** combined with **Naive Bayes** can deliver strong baseline performance for sentiment analysis tasks on movie review data.

While the model performs well and generalizes effectively, it is inherently limited by its lack of semantic and contextual understanding. Nonetheless, it provides a solid, interpretable foundation and a reliable benchmark before transitioning to more advanced models.



