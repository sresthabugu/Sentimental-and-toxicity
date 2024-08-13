# **Sentiment and Toxicity Analysis with IndicBERT**

This repository contains the implementation of sentiment analysis and toxicity detection using the IndicBERT model from Hugging Face. The project focuses on analyzing text data in Hindi, leveraging advanced NLP techniques to classify text as positive, negative, or neutral and to calculate a toxicity score.


## **Project Overview**

The aim of this project is to perform sentiment analysis and toxicity detection on Hindi text data. The key objectives include:
- **Sentiment Analysis:** Classify text as positive, negative, or neutral.
- **Toxicity Detection:** Compute a toxicity score for each piece of text, indicating the intensity of negative sentiment.

The project uses the **IndicBERT** model for both tasks. The final model is fine-tuned and tested on a custom Hindi dataset, providing accurate results with high precision.

## **Models Used**

Two primary models were considered for sentiment analysis:
- **IndicBERT:** A transformer-based model pre-trained on large-scale Indian language datasets. It has been fine-tuned specifically for this project to classify sentiments in Hindi.
- **Glot500:** Another model that was evaluated for sentiment analysis, though **IndicBERT** was ultimately chosen due to its superior performance on the dataset.

## **Data Preparation**

The dataset used in this project consists of 1,000 samples of Hindi text, each labeled with a sentiment (positive, negative, neutral). The data was preprocessed, tokenized, and split into training, validation, and test sets.

Key steps in data preparation:
1. **Data Loading:** The dataset is read from an ODS file.
2. **Label Encoding:** Sentiment labels are converted to numerical form.
3. **Data Splitting:** The dataset is split into training (64%), validation (16%), and test (20%) sets.
4. **Tokenization:** The text is tokenized using the IndicBERT tokenizer.

## **Training and Evaluation**

The model was trained using the Hugging Face `Trainer` class. Training parameters included:
- **Epochs:** 15
- **Learning Rate:** 2e-5
- **Batch Size:** 8
- **Evaluation Strategy:** Per epoch

During training, the model's performance was evaluated on the validation set. Metrics such as accuracy, F1 score, precision, recall, and confusion matrix were computed.

## **Results**

The final model achieved:
- **Accuracy:** 84.38%
- **F1 Macro Score:** 0.835
- **F1 Micro Score:** 0.843
- **Precision:** 0.843
- **Recall:** 0.843

These results demonstrate that the model performs well in classifying sentiments and calculating toxicity scores, making it a robust tool for NLP tasks in Hindi.

