# SMS_SpamClassification
Introduction
This project is a machine learning model that classifies SMS messages as either spam or ham (not spam). It uses Natural Language Processing (NLP) techniques and a Naïve Bayes classifier to detect spam messages.

Features
✅ Preprocesses SMS messages (text cleaning, stopword removal, lemmatization)
✅ Extracts features using TF-IDF Vectorization
✅ Trains a Multinomial Naïve Bayes model
✅ Predicts whether an SMS is spam or ham
✅ Visualizes data insights using Matplotlib and Seaborn

Dataset
The dataset used is newsmsspam.csv, which contains labeled SMS messages.
Labels:
1 → Spam
0 → Ham
Installation & Setup
Requirements
Make sure you have Python installed, then install the required libraries:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn nltk scikit-learn
Run the Script
Clone the repository and run the script:

bash
Copy
Edit
git clone https://github.com/your-username/sms-spam-classification.git
cd sms-spam-classification
python spam_classifier.py
Project Workflow
Load and preprocess the dataset
Perform exploratory data analysis (EDA)
Feature engineering (word count, presence of currency symbols, numbers)
Train the Naïve Bayes model
Predict whether a new SMS is spam or ham
Example Usage
You can test a sample message using the predict_spam() function:

python
Copy
Edit
sample_message = "Win a brand new iPhone! Click the link below to claim your prize."
if predict_spam(sample_message):
    print("Spam detected!")
else:
    print("This is a normal message.")
Results
Model Accuracy: (Add your accuracy score here, e.g., 95%)
Confusion Matrix Visualization: (Generated using Seaborn)
Future Enhancements
🔹 Improve preprocessing with advanced text cleaning techniques
🔹 Implement deep learning models (LSTM, BERT) for better classification
🔹 Deploy the model as a Flask API or Web App

Contributors
👤 Your Name (@your-github-handle)

License
This project is licensed under the MIT License.
