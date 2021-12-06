# 1. Data Introduction
The dataset is news headline date download from Kaggle https://www.kaggle.com/setseries/news-category-dataset.

It contains 9 news categories: BUSINESS, ENTERTAINMENT, FOOD & DRINK, PARENTING, POLITICS, SPORTS, STYLE & BEAUTY, TRAVEL and WORLD NEWS, consisting of 44,910 news headlines.

Preview of dataset:
![image](https://user-images.githubusercontent.com/89607189/144772167-20e9c53f-b684-4fa2-8131-1c7330c2fcd3.png)

The data contains nine different categories:
![image](https://user-images.githubusercontent.com/89607189/144772236-5748adb4-4ef0-4615-adbf-cca9aca91e04.png)

# 2. Data Cleaning

### 3.5 Bert
BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. 
set the bach size is 15, and epoch is 2, Create batched inputs using Huggingface's DistilBERT tokenizer
### 3.6 LSTM
Long short term memory is a special RNN model, which is effective in memorizing important information. We could find that other non-neural network classification techniques they are trained on multiple word as separate inputs that are just word having no actual meaning as a sentence, and while predicting the class it will give the output according to statistics and not according to meaning. That means, every single word is classified into one of the categories.However, in LSTM we can use a multiple word string to find out the class to which it belongs. This is very helpful while working with Natural language processing. If we use appropriate layers of embedding and encoding in LSTM, the model will be able to find out the actual meaning in input string and will give the most accurate output class. It can also mainly to solve the problems of gradient disappearance and gradient explosion in the process of long sequence training. In short, LSTM usually can perform better in longer sequences than ordinary RNN and other non-neural network classification techniques.
