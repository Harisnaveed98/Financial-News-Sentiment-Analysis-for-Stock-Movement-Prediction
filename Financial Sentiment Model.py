"""
Financial News Sentiment Analysis for Stock Movement Prediction
A compact deep learning model that analyzes financial news headlines to predict sentiment

Author: [Your Name]
Dataset: Financial PhraseBank (download from Kaggle)
URL: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
Place the downloaded CSV as 'financial_news.csv' in the same directory as this script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os, sys
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data(file_path='financial_news.csv'):
    """Load and prepare the financial news dataset"""
    print("Loading dataset...")
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: Dataset file '{file_path}' not found!")
        print("\nPlease download the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news")
        print("2. Download the CSV file")
        print("3. Rename it to 'financial_news.csv' and place in this directory\n")
        sys.exit(1)
    
    # The dataset has no header, so assign correct names
    df = pd.read_csv(file_path, header=None, names=['Sentiment', 'News'], encoding='latin-1')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSentiment distribution:\n{df['Sentiment'].value_counts()}")
    
    # Drop neutral samples, keep only positive & negative
    df = df[df['Sentiment'] != 'neutral']
    
    # Map labels: positive -> 1, negative -> 0
    df['label'] = df['Sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    return df['News'].values, df['label'].values


def prepare_sequences(texts, labels, max_words=5000, max_len=50):
    """Tokenize and pad text sequences"""
    print("\nTokenizing texts...")
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Sequence shape: {padded_sequences.shape}")
    
    return padded_sequences, labels, tokenizer


def build_model(max_words=5000, max_len=50, embedding_dim=128):
    """Construct the BiLSTM neural network"""
    print("\nBuilding model...")
    
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    print(model.summary())
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    """Train the model with callbacks"""
    print("\nTraining model...")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                               restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                 patience=2, min_lr=1e-6, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=epochs, batch_size=batch_size,
                       callbacks=[early_stop, reduce_lr], verbose=1)
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"{'='*70}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return y_pred, y_pred_proba


def visualize_results(history, y_test, y_pred, y_pred_proba):
    """Create visualizations of training and results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Financial Sentiment Analysis - Model Performance', fontsize=16, fontweight='bold')
    
    # Training accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Accuracy Over Epochs', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss Over Epochs', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
               xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve', fontweight='bold')
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'sentiment_analysis_results.png'")
    plt.show()


def predict_sentiment(model, tokenizer, text, max_len=50):
    """Predict sentiment for a new headline"""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "Positive 📈" if prediction > 0.5 else "Negative 📉"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence


def main():
    """Main execution pipeline"""
    print("="*70)
    print("Financial News Sentiment Analysis with Deep Learning")
    print("="*70)
    
    # Hyperparameters
    MAX_WORDS = 5000
    MAX_LEN = 50
    EMBEDDING_DIM = 128
    EPOCHS = 20
    BATCH_SIZE = 32
    
    # Load data
    texts, labels = load_and_preprocess_data('financial_news.csv')
    
    # Prepare sequences
    X, y, tokenizer = prepare_sequences(texts, labels, MAX_WORDS, MAX_LEN)
    
    # Split data (70% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, 
                                                         random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                     random_state=42, stratify=y_temp)
    
    print(f"\nData splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Build and train
    model = build_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM)
    history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)
    
    # Evaluate
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Visualize
    visualize_results(history, y_test, y_pred, y_pred_proba)
    
    # Save model
    model.save('sentiment_model.h5')
    print("\n✓ Model saved as 'sentiment_model.h5'")
    
    # Demo predictions
    print("\n" + "="*70)
    print("Sample Predictions on New Headlines:")
    print("="*70)
    
    sample_headlines = [
        "Company reports record profits, stock surges to all-time high",
        "Major layoffs announced as company struggles with declining revenue",
        "Merger deal approved, shareholders celebrate overwhelming success",
        "CEO resigns amid fraud scandal, shares plummet after hours",
        "Strong quarterly earnings beat analyst expectations significantly"
    ]
    
    for headline in sample_headlines:
        sentiment, confidence = predict_sentiment(model, tokenizer, headline, MAX_LEN)
        print(f"\n📰 {headline}")
        print(f"   → {sentiment} (Confidence: {confidence:.1%})")
    
    print("\n" + "="*70)
    print("✓ Training complete! Check 'sentiment_analysis_results.png' for visualizations")
    print("="*70)


if __name__ == "__main__":
    main()
