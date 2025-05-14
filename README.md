# Hespress Comments Sentiment Analysis Model

This repository contains a deep learning model for Arabic sentiment analysis, specifically trained on comments from Hespress (a Moroccan news portal) and other Arabic social media sources.

## Repository Contents

- **H_notebook.ipynb**: Jupyter notebook containing the complete modeling process, including data preprocessing, model training, evaluation, and optimization
- **model.keras**: Trained deep learning model for Arabic sentiment classification
- **tokenizer.json**: Tokenizer for preprocessing Arabic text before feeding it to the model
- **data/**: Directory containing the training and testing datasets

## Model Overview

The sentiment analysis model classifies Arabic text into three categories:
- **Positive** (2)
- **Negative** (0)
- **Neutral** (1)

## Data Sources

The model was trained using two main datasets:
1. **Tweet Sentiment Multilingual Dataset**: [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
2. **DZ Sentiment YouTube Comments Dataset**: [Abdou/dz-sentiment-yt-comments](https://huggingface.co/datasets/Abdou/dz-sentiment-yt-comments)



## Requirements
```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.19.2
pandas>=1.1.5
```

## Model Performance

The model achieves the following performance metrics on the test set:
- **Accuracy**: ~92%
- **F1 Score**: ~91%
- **Precision**: ~91%
- **Recall**: ~90%
