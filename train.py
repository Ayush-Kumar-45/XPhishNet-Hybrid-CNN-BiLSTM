#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XPhishNet: Intelligent Phishing Detection
Training pipeline with CNN + LSTM + Attention and behavioural features.
"""
import argparse
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import nltk

from src.data_loader import load_datasets, combine_and_clean
from src.preprocess import clean_text
from src.features import extract_behavioral_features, FEATURE_NAMES
from src.model import create_model
from src.utils import plot_history, plot_confusion_matrix

# Constants
SEED = 42
VOCAB_SIZE = 30000
MAX_SEQUENCE_LENGTH = 350
OOV_TOKEN = "<OOV>"
BATCH_SIZE = 256
EPOCHS = 10
TEST_SIZE = 0.2

def main(input_dir='.'):
    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    print("[1] Loading datasets...")
    url_df, email_df = load_datasets(input_dir)
    combined_df = combine_and_clean(url_df, email_df)
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Class distribution:\n{combined_df['label'].value_counts()}")

    print("[2] Cleaning and feature extraction...")
    combined_df['clean_text'] = combined_df['text'].apply(clean_text)
    beh_df = combined_df['text'].apply(extract_behavioral_features).apply(pd.Series)
    combined_df = pd.concat([combined_df, beh_df], axis=1)

    # Tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(combined_df['clean_text'])
    sequences = tokenizer.texts_to_sequences(combined_df['clean_text'])
    X_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    X_beh = combined_df[FEATURE_NAMES].values
    y = combined_df['label'].values

    # Scale behavioural features
    scaler = StandardScaler()
    X_beh_scaled = scaler.fit_transform(X_beh)

    # Train/validation split
    X_text_train, X_text_val, X_beh_train, X_beh_val, y_train, y_val = train_test_split(
        X_text, X_beh_scaled, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # Dataset pipelines
    train_ds = tf.data.Dataset.from_tensor_slices(((X_text_train, X_beh_train), y_train))
    train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((X_text_val, X_beh_val), y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    print("[3] Building model...")
    model = create_model(VOCAB_SIZE, MAX_SEQUENCE_LENGTH, X_beh_scaled.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1)
    ]

    print("[4] Training...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Load best model
    model = tf.keras.models.load_model('best_model.keras')
    print("Loaded best model.")

    print("[5] Evaluation...")
    y_pred_proba = model.predict(val_ds)
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_val, y_pred_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=['Legitimate','Phishing']))

    plot_confusion_matrix(y_val, y_pred)
    plot_history(history)

    # Save artifacts
    model.save('phishing_cnn_lstm_attention.keras')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model, tokenizer and scaler saved.")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XPhishNet phishing detector.')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Directory containing CSV datasets (recursive search)')
    args = parser.parse_args()
    main(args.input_dir)