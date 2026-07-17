#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XPhishNet: Real‑time prediction script.
Usage: python predict.py "Your suspicious text here"
"""
import sys
import numpy as np
import tensorflow as tf
import pickle
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts
model = tf.keras.models.load_model('phishing_cnn_lstm_attention.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Preprocessing functions (copied from training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

URGENT_KEYWORDS = ['urgent','verify','password','account','bank','click','login','security','limited','winner','update','confirm']
SUSPICIOUS_KEYWORDS = ['free','prize','win','money','wire','transfer','credit','card','access','unauthorized','alert','suspend','locked']

FEATURE_NAMES = ['url_length','num_dots','num_digits','num_special','has_ip','has_https',
                 'suspicious_keywords_count','domain_length','num_subdomains','domain_depth',
                 'num_slashes','num_question','num_equal','num_ampersand','num_hyphens',
                 'num_underscores','ratio_digits','email_length','num_urls',
                 'num_exclamation','num_uppercase','suspicious_word_count','urgent_word_count',
                 'urgency_score','entropy']

def extract_behavioral_features(text):
    features = {}
    s = str(text)
    features['url_length'] = len(s)
    features['num_dots'] = s.count('.')
    features['num_digits'] = sum(c.isdigit() for c in s)
    features['num_special'] = sum(not c.isalnum() and not c.isspace() for c in s)
    features['has_ip'] = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', s) else 0
    features['has_https'] = 1 if 'https' in s.lower() else 0
    features['suspicious_keywords_count'] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in s.lower())
    match = re.search(r'(?:https?://)?([^/]+)', s)
    if match:
        domain = match.group(1)
        features['domain_length'] = len(domain)
        features['num_subdomains'] = domain.count('.')
        features['domain_depth'] = len(domain.split('.'))
    else:
        features['domain_length'] = 0
        features['num_subdomains'] = 0
        features['domain_depth'] = 0
    features['num_slashes'] = s.count('/')
    features['num_question'] = s.count('?')
    features['num_equal'] = s.count('=')
    features['num_ampersand'] = s.count('&')
    features['num_hyphens'] = s.count('-')
    features['num_underscores'] = s.count('_')
    features['ratio_digits'] = features['num_digits'] / max(1, len(s))
    features['email_length'] = len(s)
    features['num_urls'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', s))
    features['num_exclamation'] = s.count('!')
    features['num_uppercase'] = sum(1 for c in s if c.isupper())
    features['suspicious_word_count'] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in s.lower())
    features['urgent_word_count'] = sum(1 for kw in URGENT_KEYWORDS if kw in s.lower())
    features['urgency_score'] = features['urgent_word_count'] + 0.5 * features['num_exclamation']
    char_counts = {}
    for ch in s:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    prob = [cnt / len(s) for cnt in char_counts.values()] if len(s) > 0 else [0]
    features['entropy'] = -sum(p * np.log2(p) for p in prob if p > 0) if len(s) > 0 else 0
    return features

MAX_SEQUENCE_LENGTH = 350

def predict_phishing(text):
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    beh_dict = extract_behavioral_features(text)
    beh_values = np.array([beh_dict[f] for f in FEATURE_NAMES]).reshape(1, -1)
    beh_scaled = scaler.transform(beh_values)
    prob = model.predict([pad, beh_scaled], verbose=0)[0][0]
    pred = 'Phishing' if prob >= 0.5 else 'Legitimate'
    return pred, float(prob)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        pred, prob = predict_phishing(text)
        print(f"Text: {text}")
        print(f"Prediction: {pred} (confidence: {prob:.4f})")
    else:
        print("Usage: python predict.py \"Your suspicious text here\"")