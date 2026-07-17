# 🧠 XPhishNet: Intelligent Phishing Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![Keras](https://img.shields.io/badge/Keras-2.10+-red)
![Accuracy](https://img.shields.io/badge/Accuracy-99.15%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

**A hybrid deep learning framework for real‑time phishing detection in emails, URLs, and messages, combining CNN, BiLSTM, custom attention, and 25 behavioural features.**

[📖 Documentation](#) •
[🐛 Report Bug](https://github.com/yourusername/xphishnet/issues) •
[✨ Request Feature](https://github.com/yourusername/xphishnet/issues)

</div>

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset & Model](#-dataset--model)
- [Team & Contributions](#-team--contributions)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Deployment](#-deployment)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 About The Project

Phishing attacks are one of the most prevalent cybersecurity threats where attackers create deceptive websites, emails, or messages mimicking legitimate entities to steal sensitive information such as passwords, banking credentials, and personal data.

**XPhishNet** provides a robust solution by:

- **Ingesting** text content (emails, URLs, chat messages)
- **Extracting** 25 behavioural features (entropy, urgency score, URL structure, etc.)
- **Learning** semantic patterns via a **CNN‑LSTM‑Attention** backbone
- **Explaining** predictions using SHAP, making it suitable for security operations
- **Achieving** 99.15% accuracy on a balanced dataset

### Purpose
Empower users and organisations to detect phishing attempts in real time, reducing the risk of data breaches and financial loss.

### Scope
The system is designed to work with any textual content – emails, URLs, SMS, chat messages – and can be integrated into security platforms via a simple CLI or REST API.

### Workflow
1. **User provides text** (email body, URL, message) via CLI or API.
2. **Preprocessing** – Clean, lemmatize, remove noise.
3. **Feature extraction** – 25 behavioural features are computed.
4. **Deep learning model** – CNN + BiLSTM + Attention processes text; behavioural features are concatenated.
5. **Prediction** – Binary classification (Phishing / Legitimate) with confidence score.
6. **Explanation** – SHAP values highlight contributing features (in companion notebook).

---

## 💻 Tech Stack

### Machine Learning / Deep Learning
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Explainability
![SHAP](https://img.shields.io/badge/SHAP-9B4DCA?style=for-the-badge&logo=shap&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

### Deployment / Production
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) *(optional)*
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) *(optional)*
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

## 📊 Dataset & Model

### Dataset Statistics
| Attribute | Value |
|-----------|-------|
| **Total Samples** | 267,996 |
| **Classes** | Balanced (legitimate / phishing) |
| **Features** | 25 behavioural + text tokens |
| **Source** | Multiple CSV datasets (URLs + emails) |

### Model Specifications
| Parameter | Value |
|-----------|-------|
| *Architecture* | CNN + BiLSTM + Attention |
| *Accuracy* | 99.15% |
| *Precision* | 98.61% |
| *Recall* | 99.87% |
| *F1-Score* | 99.24% |
| *ROC-AUC* | 99.64% |
| *Text Vocabulary* | 30,000 words |
| *Max Sequence Length* | 350 tokens |
| *Behavioural Features* | 25 (entropy, urgency, URL structure, etc.) |
| *Regularisation* | L2, Dropout, BatchNorm |

### Feature Importance (Behavioural)
| Feature | Impact |
|---------|--------|
| `has_https` | High |
| `num_slashes` | High |
| `ratio_digits` | Medium |
| `entropy` | Medium |
| `urgency_score` | Medium |
| `num_hyphens` | Medium |



## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip
- Virtual environment (recommended)
- NLTK data (downloaded automatically during training)

### Step-by-Step Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/xphishnet.git
cd xphishnet

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if not done automatically)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
