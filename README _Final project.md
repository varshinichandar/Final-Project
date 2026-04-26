# Final Project - Deep Learning and Data Analysis

## Project 1: Pneumonia Detection from Chest X-Ray Images using CNN

This project builds an image classification model using a Convolutional Neural Network to detect pneumonia from chest X-ray images. The dataset used is the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which contains labeled images split into train, validation and test folders with two classes - NORMAL and PNEUMONIA.

The images are preprocessed by resizing to 150x150, normalizing pixel values and applying data augmentation on the training set. The CNN model has 3 convolutional blocks followed by fully connected layers with dropout to prevent overfitting. Class weights are used during training to handle the imbalance between normal and pneumonia samples. The model is evaluated using accuracy, precision, recall, F1 score and a confusion matrix. It achieves a test accuracy of 85.58% with a pneumonia recall of 97%.

**Dataset:** Chest X-Ray Images (Pneumonia) - Kaggle  
**Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn  
**Model saved as:** `pneumonia_cnn_model.h5`

---

## Project 2: Next Word Prediction Using Recurrent Neural Networks

This project builds a language model using an LSTM network to predict the next word in a sequence. The WikiText-2 dataset is downloaded directly using urllib and the first 500 lines are used for training to keep memory usage manageable.

The text is tokenized and converted into n-gram sequences of length 10. These sequences are padded and split into input features and target words. A data generator is used to feed batches during training. The model uses an embedding layer to convert words into vectors, a single LSTM layer to learn sequence patterns, and dense layers for classification. It is trained for 10 epochs and evaluated on a validation set. A text generation function is included to demonstrate the model predicting the next words given a seed phrase.

**Dataset:** WikiText-2 (downloaded automatically via urllib)  
**Libraries:** TensorFlow, Keras, NumPy, Matplotlib  
**Model saved as:** `next_word_lstm_model.h5`

---

## Project 3: Exploratory Data Analysis on Global Superstore Sales Dataset

This project performs a comprehensive EDA on the Global Superstore sales dataset to uncover patterns in sales, profit and customer behavior across regions and product categories.

The analysis covers data loading and cleaning, univariate analysis of sales distribution and category counts, bivariate analysis of sales vs profit by region, multivariate analysis using a correlation matrix, and time series analysis of monthly sales trends from 2014 to 2017. An interactive plotly chart is used for the time series visualization. Key findings include consistent Q4 sales peaks, mid-year dips in June-July, an overall upward sales trend over the 4 years, and the observation that higher discounts often lead to negative profit.

**Dataset:** Sample_superstore.csv  
**Libraries:** Pandas, Matplotlib, Seaborn, Plotly  
**Output:** `cleaned_superstore_data.csv`

---

## How to Run

1. Clone or download the repository
2. Install dependencies for each project as mentioned in the first cell of each notebook
3. Place the required dataset files in the same folder as the notebook
4. Run all cells from top to bottom

## Folder Structure

```
Final-Project/
├── ChestXRay.ipynb
├── Next_word_prediction.ipynb
├── Global_superstore.ipynb
├── pneumonia_cnn_model.h5
├── next_word_lstm_model.h5
├── Sample_superstore.csv
├── cleaned_superstore_data.csv
└── README.md
```
