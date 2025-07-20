#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd # Loads and handles data using Pandas.


# In[3]:


# Load the data
df = pd.read_excel(r"D:\MSC\Final Project\training_reviews01.xlsx")


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df1=df[['Review_Text','Rating']]


# In[7]:


df1.head()


# In[8]:


# Function to convert rating to sentiment
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"


# In[9]:


# Apply the function to the "Rating" column
df1["Sentiment"] = df1["Rating"].apply(rating_to_sentiment)


# In[10]:


df1.head()


# In[11]:


df3=df1.drop(['Rating'],axis=1)


# In[12]:


df3.head()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib




# Features (X) and target (y)
X = df3["Review_Text"]
y = df3["Sentiment"]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Create a pipeline with TF-IDF vectorization and Naive Bayes model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to numeric features using TF-IDF
    ('model', MultinomialNB())  # Naive Bayes classifier
])

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))


# In[14]:


import os

# Create folder if it does not exist
folder_path = r'D:\ML_AI_projects\NLP\Bank_Pulse_application\sentiment_dashboard'
os.makedirs(folder_path, exist_ok=True)

# Now save the model
save_path = os.path.join(folder_path, 'sentiment_model.pkl')

joblib.dump(pipeline, save_path)

print("✅ Model saved successfully!")


# In[15]:


from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultinomialNB())
])

scores = cross_val_score(
    pipeline, X, y,
    cv=5,
    scoring='f1_macro'
)
print("5-fold CV F1-macro scores:", scores)
print("Mean F1-macro:", scores.mean())


# In[16]:


from collections import Counter
import numpy as np

# Fit TF-IDF on the entire corpus
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)
terms = np.array(tfidf.get_feature_names_out())

# Separate by class
class_terms = {}
for label in ["Negative", "Neutral", "Positive"]:
    # Compute average TF-IDF score per term for docs in this class
    idxs = (y == label)
    avg_tfidf = X_tfidf[idxs].mean(axis=0).A1
    # Take top 20 terms
    top_idxs = np.argsort(avg_tfidf)[-20:]
    class_terms[label] = terms[top_idxs][::-1]

for label, top_terms in class_terms.items():
    print(f"\nTop terms for {label}:", top_terms)


# In[17]:


import re
issue_keywords = {
    "Service-related": [
        "service", "response", "downtime", "helpline", "communication",
        "support", "helpdesk", "issue", "delay"
    ],
    "Branch-related": [
        "branch", "branches", "location", "atm", "cash machine", "teller",
        "queue", "line", "waiting", "counter"
    ],
    "Staff-related": [
        "staff", "representative", "employee", "manager", "team", "teller",
        "officer", "agent"
    ],
    "Misbehavior": [
        "rude", "unhelpful", "misbehave", "incompetent", "argue",
        "aggressive", "impolite", "insult"
    ],
    "Charges": [
        "charges", "fee", "fees", "penalty", "interest"
    ],
    "Software-Related": [
        "downtime", "bug", "crash", "error", "login", "update", "app",
        "mobile", "website"
    ],
    "ATM-related": [
        "atm", "cash machine", "atm’s", "atm,", "atm."
    ],
}

def categorize_review(text):
    text_lower = text.lower()
    scores = {}
    for cat, kws in issue_keywords.items():
        scores[cat] = sum(bool(re.search(rf"\b{k}\b", text_lower)) for k in kws)
    best_cat, best_score = max(scores.items(), key=lambda x: x[1])
    return best_cat if best_score > 0 else "Other"

# Apply to the DataFrame
df["Issue_Category"] = df["Review_Text"].apply(categorize_review)

# Check counts
print(df["Issue_Category"].value_counts())


# In[ ]:





# In[ ]:




