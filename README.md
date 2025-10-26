# Tweet Sentiment Analysis
A Machine Learning project that classifies tweets as **Positive** or **Negative** using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).  
The project uses **TF-IDF vectorization** and **Logistic Regression** to perform sentiment classification.  
It was developed in **Google Colab** using **Python**, **NLTK**, and **Scikit-learn**.

# 🧠 Project Overview
This project demonstrates how text preprocessing and machine learning can be used to understand public sentiment on social media.  
It involves:
- Cleaning and preprocessing text data (removing symbols, converting to lowercase, stemming, etc.)
- Converting text into numerical features using TF-IDF
- Training a Logistic Regression model for binary classification
- Evaluating the model and making predictions on new tweets

# 📂 Dataset
- **Name:** Sentiment140 Dataset  
- **Source:** [Kaggle - kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- **Description:** Contains 1.6 million tweets labeled as:
  - `0` → Negative sentiment  
  - `4` → Positive sentiment  

# ⚙️ Technologies Used
- Python
- Google Colab
- NumPy, Pandas
- NLTK (Natural Language Toolkit)
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Pickle (for saving trained models)

# 🚀 Steps in the Project

**💾 Dataset Fetching**
!kaggle datasets download -d kazanova/sentiment140

**🧹 Data Cleaning & Preprocessing**
- Removed URLs, mentions, hashtags, and special characters  
- Converted text to lowercase  
- Tokenized and stemmed words using **NLTK**

**🔠 Feature Extraction**
- Transformed cleaned tweets into numerical form using **TF-IDF Vectorization**

**🧩 Model Training**
- Trained a **Logistic Regression** model on labeled tweet data  
- Split dataset into **training and test sets** for performance evaluation

**📈 Evaluation**
- Checked **accuracy, precision, recall, and F1-score**  
- Analyzed **confusion matrix** for detailed insights

**🤖 Prediction**
- Tested the model on custom input tweets to predict sentiment in real time


# 📊 Results
The trained model achieved **high accuracy** and effectively distinguishes between positive and negative tweets.  
Its **simplicity, speed, and interpretability** make it an ideal baseline for more advanced models like **LSTMs** or **Transformers** in future work.


# ❤️ Acknowledgments
Thanks to the creators of the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
and the open-source libraries that made this project possible.

