import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import os

fake_df = pd.read_csv('../data/Fake.csv')
true_df = pd.read_csv('../data/True.csv')

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vect, y_train)

score = model.score(X_test_vect, y_test)
print(f"Model Accuracy: {score:.2f}")

os.makedirs("../models", exist_ok=True)
with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
