# Sentiment Analysis 

This repository contains an implementation of a sentiment analysis model using bidirectional Long Short-Term Memory (LSTM) networks with TensorFlow and Keras. The model is designed to classify text data into positive, neutral and negative sentiments.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Video](#video)

## Overview

Sentiment analysis, or opinion mining, involves determining the sentiment expressed in a piece of text. This project utilizes an LSTM-based neural network to perform sentiment classification, leveraging the capabilities of TensorFlow and Keras for model building and training.

## Dataset

The dataset used for training and evaluation is located in the `data` directory. It consists of text samples labeled as positive, neutral or negative, suitable for sentiment classification tasks.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/farukaplan/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main implementation is provided in the Jupyter Notebook `sentiment_analysis.ipynb`. To run the notebook:

1. **Ensure Jupyter is installed**:

   ```bash
   pip install jupyter
   ```

2. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

3. **Open and execute** the `sentiment_analysis.ipynb` notebook to train and evaluate the model.

## Model Architecture

The sentiment analysis model employs an LSTM-based architecture, which is effective for processing sequential data such as text. The model comprises the following layers:

- **Embedding Layer**: Converts input words into dense vectors of fixed size.
- **Bidirectional LSTM Layer**: Captures temporal dependencies in the text data.
- **Dropout Layer**: Prevents overfitting
- **Dense Layer**: Fully connected layer with a ReLu activation function.
- **Output Layer**: Final output layer with a Softmax.

## Results

The model's performance is evaluated using accuracy and loss metrics. Detailed results, including training and validation accuracy and loss curves, are documented in the `sentiment-analysis-report.pdf` file.

## Contributing

Contributions to enhance the project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## Video
If you prefer to watch video to understand the code, you can visit following YouTube video: https://youtu.be/o95-X_zDRkU
