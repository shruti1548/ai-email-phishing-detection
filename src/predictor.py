import os
import joblib

# this file is inside the src folder, so the model files are in the same folder
folder = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(folder, 'model.pkl'))
vectorizer = joblib.load(os.path.join(folder, 'vectorizer.pkl'))

# same stopwords list as in ml.py
stopwords_list = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'on', 'off',
'and', 'or', 'but', 'if', 'this', 'that', 'i', 'me', 'my', 'you', 'your',
'we', 'our', 'he', 'she', 'it', 'they', 'them', 'his', 'her',
'have', 'has', 'had', 'do', 'does', 'did', 'so', 'as']


def clean_email_text(email_text):
    email = str(email_text)
    email = email.replace('\\n', ' ')
    email = email.replace('\n', ' ')
    email = email.replace('Subject:', ' ')
    email = email.lower()

    # keep only letters and spaces
    new_email = ""
    for letter in email:
        if letter.isalpha():
            new_email = new_email + letter
        elif letter == ' ':
            new_email = new_email + letter

    # remove stopwords
    kept_words = []
    for word in new_email.split():
        if word not in stopwords_list:
            kept_words.append(word)
    return " ".join(kept_words)


def predict_phishing(email_text):
    cleaned = clean_email_text(email_text)
    vector = vectorizer.transform([cleaned])
    probability = model.predict_proba(vector)[0][1]    # chance of phishing
    risk_score = int(round(probability * 100))
    if probability >= 0.5:
        label = "Phishing"
    else:
        label = "Legitimate"
    return label, risk_score