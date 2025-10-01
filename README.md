# Financial News Sentiment Analysis 📈📉

A deep learning model that analyzes financial news headlines to predict market sentiment using Bidirectional LSTM networks.

## Overview

This project demonstrates practical NLP and deep learning for financial text analysis. The model classifies news headlines as positive or negative sentiment, which can help investors gauge market sentiment quickly.

**Why this project?**
- Applies deep learning to real-world financial domain
- Uses BiLSTM for context-aware text understanding
- Small dataset (~50MB) but production-quality code
- Complete ML pipeline: preprocessing → training → evaluation

## Model Architecture

```
Embedding (128-dim) → BiLSTM (64) → Dropout → BiLSTM (32) → Dropout → Dense (32) → Output (Sigmoid)
```

- **BiLSTM**: Captures context from both directions in text
- **Dropout**: Prevents overfitting (0.2-0.3 rates)
- **Binary Classification**: Positive/Negative sentiment

## Dataset

**Financial PhraseBank** - ~4,840 financial news sentences with sentiment labels

**Download:**
1. Visit: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
2. Download the CSV file
3. Rename to `financial_news.csv` and place in project directory

**Dataset format:**
```
Sentiment,Sentence
positive,"Company reports record quarterly earnings"
negative,"Stock plummets after disappointing results"
```

## Installation

```bash
# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

# Place dataset
# financial_news.csv should be in the same directory as the script
```

## Usage

Simply run:
```bash
python sentiment_analysis.py
```

The script will:
1. ✅ Load and preprocess the dataset
2. ✅ Build the BiLSTM model
3. ✅ Train with early stopping (~5-10 min on CPU)
4. ✅ Evaluate on test set
5. ✅ Generate visualizations
6. ✅ Save trained model
7. ✅ Demo predictions on sample headlines

## Results

Expected performance:
- **Accuracy**: 85-90%
- **AUC**: 0.92-0.95

Output files:
- `sentiment_model.h5` - Trained model
- `sentiment_analysis_results.png` - Performance visualizations

## Code Structure

The single file contains:
- `load_and_preprocess_data()` - Data loading with error handling
- `prepare_sequences()` - Tokenization and padding
- `build_model()` - BiLSTM architecture definition
- `train_model()` - Training with callbacks
- `evaluate_model()` - Metrics calculation
- `visualize_results()` - 4 plots (accuracy, loss, confusion matrix, ROC)
- `predict_sentiment()` - Inference on new text
- `main()` - Complete pipeline orchestration

## Example Predictions

```
📰 Company reports record profits, stock surges to all-time high
   → Positive 📈 (Confidence: 94.2%)

📰 Major layoffs announced as company struggles with declining revenue
   → Negative 📉 (Confidence: 89.7%)
```

## Customization

Adjust hyperparameters in `main()`:
```python
MAX_WORDS = 5000      # Vocabulary size
MAX_LEN = 50          # Sequence length
EMBEDDING_DIM = 128   # Embedding dimensions
EPOCHS = 20           # Training epochs
BATCH_SIZE = 32       # Batch size
```

## Key Features

- ✅ Clean, modular code with clear comments
- ✅ Proper train/validation/test split (70/15/15)
- ✅ Early stopping and learning rate scheduling
- ✅ Comprehensive evaluation metrics
- ✅ Professional visualizations
- ✅ Error handling for missing dataset
- ✅ Reproducible results (fixed random seeds)

