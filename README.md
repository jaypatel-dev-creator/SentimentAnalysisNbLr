# Sentiment Analysis using Naive Bayes, Logistic Regression & TF-IDF

This project implements a text sentiment classification system using classical Natural Language Processing (NLP) techniques.
The goal is to classify text reviews as positive or negative using a TF-IDF vectorizer with Multinomial Naive Bayes and Logistic Regression classifiers.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://github.com/jaypatel-dev-creator/SentimentAnalysisNbLr/blob/main/SentimentAnalysis.ipynb)


## Project Overview

Sentiment analysis is a common NLP task used in applications such as:

* Product review analysis
* Social media monitoring
* Customer feedback systems

This project demonstrates how traditional machine learning models can be effectively applied to sentiment analysis without deep learning, while following a clean, industry-standard ML workflow and comparing multiple classifiers.

## Dataset Overview

**Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

* **Total samples:** 50,000 movie reviews
* **Positive reviews:** 25,000
* **Negative reviews:** 25,000
* Balanced binary sentiment classification dataset

---


# Model Pipeline

The workflow follows standard machine learning best practices:

## 1. Text Preprocessing

* Lowercasing
* Stopword handling
* Tokenization (handled internally by TF-IDF)

## 2. Feature Engineering

* TF-IDF Vectorization
* Unigrams + Bigrams (`ngram_range=(1,2)`)

## 3. Model Training

* Multinomial Naive Bayes
* Logistic Regression

## 4. Evaluation

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix

## 5. Prediction

* Sentiment prediction on unseen text
* Model comparison on the same input samples

## Technologies Used

* Python
* Scikit-learn
* NumPy
* Jupyter Notebook


---



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


---



# Key Learnings

* Classical machine learning models rely heavily on feature engineering
* TF-IDF captures word importance rather than semantic meaning
* Naive Bayes is fast and interpretable but assumes word independence
* Logistic Regression provides stronger decision boundaries for sentiment classification tasks
* Comparing multiple models on the same feature set improves confidence in r


---

##  Future Improvements


### Use semantic word embeddings
- Replace TF-IDF with **Word2Vec, GloVe, or FastText** to capture semantic similarity between words.
- Generate sentence-level representations using averaged or weighted embeddings.

### Adopt transformer-based models
- Experiment with **DistilBERT or BERT** to leverage contextual understanding of language.
- Fine-tune models on domain-specific sentiment data for improved performance.

### Expand the dataset
- Increase training data size to improve model generalization and reduce prediction variance.

### Improve evaluation strategy
- Apply **k-fold cross-validation** to obtain more robust and reliable performance estimates across different data splits.


---


# Conclusion

This project demonstrates that TF-IDF combined with classical machine learning models such as Naive Bayes and Logistic Regression can deliver strong baseline performance for sentiment analysis on movie review data.

While these models lack deep semantic understanding, they are fast, interpretable, and effective. Logistic Regression consistently demonstrates improved robustness over Naive Bayes, making it a strong next-step baseline before transitioning to deep learning–based solu


