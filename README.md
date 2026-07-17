# XPhishNet-Hybrid-CNN-BiLSTM

Deep learning-based phishing detection framework integrating NLP preprocessing, Hybrid CNN–BiLSTM, engineered feature extraction, and Explainable AI (SHAP) for phishing email and URL classification.

Hybrid deep learning model combining CNN, LSTM, attention, and hand-crafted behavioral features for phishing detection in emails and URLs.

## Features

- Multi-dataset ingestion (URLs, emails)
- Text cleaning and NLP preprocessing
- 25 behavioral features (entropy, urgency, URL structure, etc.)
- Custom attention mechanism
- SHAP explainability (in notebook)

## Project Structure

- `train.py` – main training script
- `predict.py` – real-time inference
- `src/` – modular helpers

## Usage

### Training

```bash
python train.py --input_dir /path/to/datasets
```
