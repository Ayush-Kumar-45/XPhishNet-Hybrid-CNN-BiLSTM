"""
Behavioural feature extraction.
"""
import re
import numpy as np

URGENT_KEYWORDS = [
    'urgent', 'verify', 'password', 'account', 'bank', 'click',
    'login', 'security', 'limited', 'winner', 'update', 'confirm'
]
SUSPICIOUS_KEYWORDS = [
    'free', 'prize', 'win', 'money', 'wire', 'transfer', 'credit',
    'card', 'access', 'unauthorized', 'alert', 'suspend', 'locked'
]

FEATURE_NAMES = [
    'url_length', 'num_dots', 'num_digits', 'num_special', 'has_ip', 'has_https',
    'suspicious_keywords_count', 'domain_length', 'num_subdomains', 'domain_depth',
    'num_slashes', 'num_question', 'num_equal', 'num_ampersand', 'num_hyphens',
    'num_underscores', 'ratio_digits', 'email_length', 'num_urls',
    'num_exclamation', 'num_uppercase', 'suspicious_word_count', 'urgent_word_count',
    'urgency_score', 'entropy'
]

def extract_behavioral_features(text):
    """Extract 25 behavioural features from raw text."""
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