# 1. Data Introduction
The dataset is news headline date download from Kaggle https://www.kaggle.com/setseries/news-category-dataset.

It contains 9 news categories: BUSINESS, ENTERTAINMENT, FOOD & DRINK, PARENTING, POLITICS, SPORTS, STYLE & BEAUTY, TRAVEL and WORLD NEWS, consisting of 44,910 news headlines.

Preview of dataset:
![image](https://user-images.githubusercontent.com/89607189/144772167-20e9c53f-b684-4fa2-8131-1c7330c2fcd3.png)

The data contains nine different categories:
![image](https://user-images.githubusercontent.com/89607189/144772236-5748adb4-4ef0-4615-adbf-cca9aca91e04.png)

# 2. Data Cleaning
Train Data news category distribution:

![image](https://user-images.githubusercontent.com/89607189/144772555-6d6d985f-9a27-4724-b094-9ba3a3b76cc6.png)

Test Data news category distribution:

![image](https://user-images.githubusercontent.com/89607189/144772575-2b74a383-b46d-4f4a-a19b-50871c322fb9.png)

# 3. Model

### 3.1 SVM

### 3.2 Multinomial Navie Bayes

### 3.3 Decision Tree

### 3.4 Random Forest

### 3.5 Bert
BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. 
set the bach size is 15, and epoch is 2, Create batched inputs using Huggingface's DistilBERT tokenizer
### 3.6 LSTM
Long short term memory is a special RNN model, which is effective in memorizing important information. We could find that other non-neural network classification techniques they are trained on multiple word as separate inputs that are just word having no actual meaning as a sentence, and while predicting the class it will give the output according to statistics and not according to meaning. That means, every single word is classified into one of the categories.However, in LSTM we can use a multiple word string to find out the class to which it belongs. This is very helpful while working with Natural language processing. If we use appropriate layers of embedding and encoding in LSTM, the model will be able to find out the actual meaning in input string and will give the most accurate output class. It can also mainly to solve the problems of gradient disappearance and gradient explosion in the process of long sequence training. In short, LSTM usually can perform better in longer sequences than ordinary RNN and other non-neural network classification techniques.

# 4. Model Evaluation

### 4.1 App Deployment
We used Streamlit, an open-source python framework, to deploy Web Apps for the models.
Our first Web App compares the Random Forest, Multinomial Naïve Bayes, and Support Vector Machine models for our test dataset. We can cross compare the predicted values as well as probability distributions.
Our second Web App enables users to input news headlines, and predicts the News headline classification and probability distributions using Multinomial Naïve Bayes model.

#### Compare Models on Test Dataset:

![image](https://github.com/zhanglu980608/NLP-Final-Project/blob/main/final_web.gif)

#### User Input News Headlines:

![image](https://github.com/zhanglu980608/NLP-Final-Project/blob/main/final_web_input_news_headline.gif)

