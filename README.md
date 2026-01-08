# 🎬 IMDB Movie Review Sentiment Analysis with Simple RNN

This project is part of a **learning-based project from a Udemy [Complete Data Science, Machine Learning, DL, NLP Bootcamp](https://www.udemy.com/certificate/UC-14b8d5ed-d5a1-4bb7-95c6-1788c0df30b9/)**.  
The objective of this project is to **build an end-to-end sentiment analysis system** using a **Simple RNN**, starting from word embedding, model training, and prediction, to final deployment with **Streamlit**.

> ⚠️ This project is **not a final capstone project**, but a guided project completed as part of a structured learning process.

---

## 📌 Project Objectives

1. Understand and implement **word embedding** using Keras.
2. Build and train a **Simple RNN model** for binary sentiment classification.
3. Perform **inference and prediction** on new movie reviews.
4. Deploy the trained model using **Streamlit** for an interactive web application.
5. Publish the application online via **Streamlit Community Cloud**.

---

## 📊 Dataset Overview

- **Dataset**: IMDB Movie Reviews  
- **Source**: Keras `imdb` dataset  
- **Size**: 25,000 training reviews, 25,000 testing reviews  
- **Task**: Binary classification (Positive / Negative)

### Preprocessing Steps
- Vocabulary size: 10,000 most frequent words
- Sequence padding to max length: 500 tokens
- Embedding dimension: 128

---

## 🧠 Model Architecture

The model follows a **Simple RNN** architecture:

```
Embedding Layer (vocab_size=10000, embedding_dim=128)
↓
SimpleRNN Layer (128 units, activation='relu')
↓
Dense Layer (1 unit, activation='sigmoid')
```

**Total Parameters**: 1,313,025 (5.01 MB)

### Training Details
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 10 (with EarlyStopping)
- **Validation Split**: 0.2

### Final Performance
- **Training Accuracy**: ~96%
- **Validation Accuracy**: ~85%

![Training Logs](images/Training%20Logs.png)

---

## 🔧 Workflow

### 1. Embedding Exploration (`embedding.ipynb`)
- One-hot encoding of sample sentences
- Padding sequences
- Embedding layer visualization
- Understanding how words are mapped to dense vectors

![OneHot Representation](images/OneHot%20Representation.png)

### 2. Model Training (`simplernn.ipynb`)
- Load and preprocess IMDB dataset
- Build and compile Simple RNN model
- Train with early stopping
- Save model to `.h5` file

![Model Summary](images/Model%20Summary.png)

### 3. Prediction & Inference (`prediction.ipynb`)
- Load pre-trained model
- Helper functions for text preprocessing and decoding
- Example sentiment prediction

![Prediction Output](images/Prediction%20Output.png)

### 4. Deployment (`main.py`)
- Interactive Streamlit app
- User input text preprocessing
- Real-time sentiment prediction

![App Preview](images/App%20Preview.png)

---

## 🌐 Live Deployment

The model is deployed online via **Streamlit Community Cloud**:

🔗 **Live App**: [https://imbdmoviereviewsentimentanalysissimplernndlproject.streamlit.app/](https://imbdmoviereviewsentimentanalysissimplernndlproject.streamlit.app/)

### Features of the Deployed App:
- User-friendly text input for movie reviews
- Real-time sentiment prediction (Positive/Negative)
- Prediction confidence score
- Clean and responsive interface

---

## 🛠️ Tech Stack

- **Language**: Python
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy
- **Web Framework**: Streamlit
- **Model Format**: H5
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git & GitHub

---

## 📚 Learning Outcomes

This project helped me understand:

- Word embedding and sequence representation in NLP
- Building and training RNN models for text classification
- Model persistence and loading in Keras
- Text preprocessing for inference
- End-to-end deployment of deep learning models with Streamlit
- Publishing applications to Streamlit Community Cloud

---

## ⚠️ Note on Model Performance

This model uses a **Simple RNN**, which is a basic recurrent architecture. While it achieves reasonable performance (~85% validation accuracy), more advanced architectures like **LSTM** or **GRU** could yield better results, especially for longer sequences. This project focuses on the foundational workflow and deployment pipeline rather than state-of-the-art accuracy.

---

**Project Completed as Part of Data Science / Deep Learning Bootcamp**  
*Guided project for educational purposes.*