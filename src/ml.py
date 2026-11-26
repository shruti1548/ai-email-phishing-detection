import pandas as pd
import re
import joblib 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load the dataset
data = pd.read_csv('phishing_legit_dataset_KD_10000.csv')

print("Shape:", data.shape)
print(data.head(2))

# Section 2: Handle missing values
print("Missing values per column:")
print(data.isnull().sum())

# Drop rows where the email text or label is missing
# (we can't train on a row with no text or no answer key)
before = len(data)
df = data.dropna(subset=['text', 'label'])
after = len(df)
print(f"Rows before: {before}, after dropping missing text/label: {after}")

# Section 3: Remove duplicates 
before = len(data)
df = df.drop_duplicates(subset=['text'])
after = len(df)
print(f"Rows before: {before}, after removing duplicate emails: {after}")
print(f"Removed {before - after} duplicate rows")

print(data.shape)

# Section 4: clean the text

# an empty list to collect the cleaned emails
cleaned_list = []

# go through each email one at a time
for email in data['text']:

    # make sure it is a string
    email = str(email)

    # replace the \n symbol with a space
    email = email.replace('\n', ' ')

    # remove the word Subject:
    email = email.replace('Subject:', ' ')

    # make everything lowercase
    email = email.lower()

    # go letter by letter and only keep letters and spaces
    new_email = ""
    for letter in email:
        if letter.isalpha():
            new_email = new_email + letter
        elif letter == ' ':
            new_email = new_email + letter

    # split and join to remove extra spaces
    words = new_email.split()
    new_email = " ".join(words)

    # add this cleaned email to our list
    cleaned_list.append(new_email)

# save the list as a new column
data['clean_text'] = cleaned_list

print(data[['text', 'clean_text']].head(3))

# Section 4: clean the text

# an empty list to collect the cleaned emails
cleaned_list = []

# go through each email one at a time
for email in data['text']:

    # make sure it is a string
    email = str(email)

    # replace the \n symbol with a space
    email = email.replace('\n', ' ')

    # remove the word Subject:
    email = email.replace('Subject:', ' ')

    # make everything lowercase
    email = email.lower()

    # go letter by letter and only keep letters and spaces
    new_email = ""
    for letter in email:
        if letter.isalpha():
            new_email = new_email + letter
        elif letter == ' ':
            new_email = new_email + letter

    # split and join to remove extra spaces
    words = new_email.split()
    new_email = " ".join(words)

    # add this cleaned email to our list
    cleaned_list.append(new_email)

# save the list as a new column
data['clean_text'] = cleaned_list

print(data[['text', 'clean_text']].head(3))

# count how many of each label we have
label_counts = data['label'].value_counts()
print(label_counts)

# label 1 = phishing, label 0 = legitimate
phishing_count = label_counts[1]
legit_count = label_counts[0]

print("phishing emails:", phishing_count)
print("legitimate emails:", legit_count)

# class distribution as percentage
print(data['label'].value_counts(normalize=True) * 100)


# check missing values in each column
print(data.isnull().sum())

# check how many duplicate emails exist
duplicate_count = data.duplicated(subset=['text']).sum()
print("duplicate emails:", duplicate_count)

# find the length of each email (number of characters)
length_list = []

for email in data['text']:
    email = str(email)
    length = len(email)
    length_list.append(length)

# add it as a new column
data['text_length'] = length_list

# overall stats
print(data['text_length'].describe())

# average length for phishing vs legitimate
print(data.groupby('label')['text_length'].mean())

import matplotlib.pyplot as plt
#  Visualization 1: class distribution 
# sort by index so 0 (legitimate) comes before 1 (phishing)
label_counts_sorted = label_counts.sort_index()

plt.figure(figsize=(6,4))
label_counts_sorted.plot(kind='bar', color=['green', 'red'])
plt.title('Number of Phishing vs Legitimate Emails')
plt.xlabel('Label (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Number of Emails')
plt.xticks(rotation=0)
plt.savefig('viz1_class_distribution.png')
plt.show()

# --- Visualization 2: text length histogram, phishing vs legitimate ---
legit_lengths = data[data['label'] == 0]['text_length']
phishing_lengths = data[data['label'] == 1]['text_length']

plt.figure(figsize=(7,5))
plt.hist(legit_lengths, bins=30, alpha=0.6, label='Legitimate', color='green')
plt.hist(phishing_lengths, bins=30, alpha=0.6, label='Phishing', color='red')
plt.title('Email Length Distribution (Phishing vs Legitimate)')
plt.xlabel('Email Length (characters)')
plt.ylabel('Number of Emails')
plt.legend()
plt.savefig('viz2_length_histogram.png')
plt.show()


#  Visualization 2: text length histogram, phishing vs legitimate ---
legit_lengths = data[data['label'] == 0]['text_length']
phishing_lengths = data[data['label'] == 1]['text_length']

plt.figure(figsize=(7,5))
plt.hist(legit_lengths, bins=30, alpha=0.6, label='Legitimate', color='green')
plt.hist(phishing_lengths, bins=30, alpha=0.6, label='Phishing', color='red')
plt.title('Email Length Distribution (Phishing vs Legitimate)')
plt.xlabel('Email Length (characters)')
plt.ylabel('Number of Emails')
plt.legend()
plt.savefig('viz2_length_histogram.png')
plt.show()


# --- Visualization 3: average text length per class ---
avg_length = data.groupby('label')['text_length'].mean()

plt.figure(figsize=(6,4))
avg_length.plot(kind='bar', color=['green', 'red'])
plt.title('Average Email Length by Class')
plt.xlabel('Label (0 = Legitimate, 1 = Phishing)')
plt.ylabel('Average Length (characters)')
plt.xticks(rotation=0)
plt.savefig('viz3_avg_length.png')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# load the cleaned dataset (already deduplicated and normalized)
data = pd.read_csv('phishing_cleaned.csv')

print(data.shape)
print(data.head(3))


# split into inputs (X) and target (y)
X = data['text']
y = data['label']

# split into training data and testing data
# test_size=0.2 means 20% of emails go to testing, 80% for training
# random_state=42 makes the split the same every time we run this code
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("training emails:", len(X_train))
print("testing emails:", len(X_test))

print("training label balance:")
print(y_train.value_counts(normalize=True))

print("testing label balance:")
print(y_test.value_counts(normalize=True))

# create the TF-IDF converter
# max_features=5000 keeps only the 5000 most useful words, to keep things simple and fast
vectorizer = TfidfVectorizer(max_features=5000)

# fit the vectorizer ONLY on training data
# this means it only learns vocabulary and word importance from training emails
X_train_tfidf = vectorizer.fit_transform(X_train)

# transform the test data using the SAME vectorizer
# we do NOT fit again here, we only apply what was already learned
X_test_tfidf = vectorizer.transform(X_test)

print("training matrix shape:", X_train_tfidf.shape)
print("testing matrix shape:", X_test_tfidf.shape)


# create the logistic regression model
model = LogisticRegression(random_state=42)

# train it using the training data only
model.fit(X_train_tfidf, y_train)

# Save the trained model and TF-IDF vectorizer

model_artifacts = {
    'model': model,
    'vectorizer': vectorizer
}




joblib.dump(model_artifacts, BASE_DIR / "model.pkl")

print("model trained successfully")




# use the trained model to predict labels for the test emails
# (the model has never seen these emails or their answers before)
y_pred = model.predict(X_test_tfidf)

print("first 10 actual labels:   ", list(y_test[:10]))
print("first 10 predicted labels:", list(y_pred[:10]))


# accuracy: percentage of emails predicted correctly overall
accuracy = accuracy_score(y_test, y_pred)

# precision: of the emails we PREDICTED as phishing, how many actually were phishing
precision = precision_score(y_test, y_pred)

# recall: of the emails that were ACTUALLY phishing, how many did we correctly catch
recall = recall_score(y_test, y_pred)

# f1-score: a balance between precision and recall
f1 = f1_score(y_test, y_pred)

print("accuracy: ", accuracy)
print("precision:", precision)
print("recall:   ", recall)
print("f1 score: ", f1)

# confusion matrix: a table showing correct vs incorrect predictions in detail
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:")
print(cm)

import os


save_folder = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(save_folder, 'model.pkl'))
joblib.dump(vectorizer, os.path.join(save_folder, 'vectorizer.pkl'))
print("saved model.pkl and vectorizer.pkl in", save_folder)



model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")





