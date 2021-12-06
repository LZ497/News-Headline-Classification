# 1. Data Introduction
The dataset is news headline date download from Kaggle https://www.kaggle.com/setseries/news-category-dataset.

It contains 9 news categories: BUSINESS, ENTERTAINMENT, FOOD & DRINK, PARENTING, POLITICS, SPORTS, STYLE & BEAUTY, TRAVEL and WORLD NEWS, consisting of 44,910 news headlines.

Preview of dataset:
![image](https://user-images.githubusercontent.com/89607189/144772167-20e9c53f-b684-4fa2-8131-1c7330c2fcd3.png)

The data contains nine different categories:

<div align='center'><img width="700" height="700" src="https://user-images.githubusercontent.com/89607189/144772236-5748adb4-4ef0-4615-adbf-cca9aca91e04.png"></div>

# 2. Data Cleaning
To increase model accuracy, the 'headline' and the short_description is combined into a new column 'text'. After tokenization, stemming and lemmatization, the clean dataset is split into training and testing dataset in a 4:1 ratio.

Train Data news category distribution:

<img width="500" height="380" src="https://user-images.githubusercontent.com/89607189/144772555-6d6d985f-9a27-4724-b094-9ba3a3b76cc6.png">

Test Data news category distribution:

<img width="500" height="380" src="https://user-images.githubusercontent.com/89607189/144772575-2b74a383-b46d-4f4a-a19b-50871c322fb9.png">

The category distribution in the train and test dataset are both well balanced.

# 3. Model
The pipeline is used to process data combining countvectorize, TFIDF and classification model.

## 3.1 Multinomial Navie Bayes
Multinomial Naive Bayes algorithm is based on the Bayes theorem. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output. The principle of it is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature.

GridSearchCV is used to find the optimal alpha of the model.

Model performance:

<img width="400" height="240" src="https://user-images.githubusercontent.com/89607189/144783598-8e5994fd-3f02-40a3-95b5-c2ac97943f55.png">

Confusion matric:

<img width="400" height="300" src="https://user-images.githubusercontent.com/89607189/144783617-219ded15-d46b-4c4c-9a8c-1de635caa78e.png">

## 3.2 Perceptron
Model performance:

<img width="400" height="240" src="https://user-images.githubusercontent.com/89607189/144783656-a00a9a8f-fd67-4014-9ce0-8ea51cd7118b.png">

Confusion matric:

<img width="400" height="300" src="https://user-images.githubusercontent.com/89607189/144783682-f57726d3-836d-4f22-8b73-1f62d4c9a521.png">

## 3.3 SVM
Support Vector Machine is a supervised machine learning algorithm,which is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.

Model performance:

<img width="400" height="240" src="https://user-images.githubusercontent.com/89607189/144783717-7167a01b-9138-4c71-80a7-a7ce829add22.png">


Confusion matric:

<img width="400" height="300" src="https://user-images.githubusercontent.com/89607189/144783732-19d4cb0e-597a-432e-bd19-de20ac67c8d7.png">

## 3.4 Decision Tree

Model performance:

<img width="400" height="240" src="https://user-images.githubusercontent.com/89607189/144783768-73c07bbf-454a-4afd-8660-d8c2bf8fc3eb.png">

Confusion matric:

<img width="400" height="300" src="https://user-images.githubusercontent.com/89607189/144783782-fbac0a43-2eb3-413a-9645-a62f35d087e6.png">

## 3.5 Random Forest
Model performance:

<img width="400" height="240" src="https://user-images.githubusercontent.com/89607189/144783810-2260c05b-ca72-4bf1-b918-3241eb3c5719.png">

Confusion matric:

<img width="400" height="300" src="https://user-images.githubusercontent.com/89607189/144783823-1c65b27b-7c62-44f8-b9ce-348c18ac4d81.png">

## 3.6 BERT
BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary.   

Set the bach size is 15, and epoch is 2, and Huggingface's DistilBERT tokenizer as our batched inputs.  

For our train datasets, the loss: 0.62453, f1-score: 0.80974, accuracy: 0.80984, precision: 0.81017, recall: 0.80979. 

For our test datasets, the loss: 0.68415, f1-score: 0.78636, accuracy: 0.78646, precision: 0.78738, recall: 0.78632. 
![image](https://github.com/zhanglu980608/NLP-Final-Project/blob/main/pics/%20bert.gif)

## 3.7 LSTM
Long short term memory is a special RNN model, which is effective in memorizing important information. We could find that other non-neural network classification techniques they are trained on multiple word as separate inputs that are just word having no actual meaning as a sentence, and while predicting the class it will give the output according to statistics and not according to meaning. That means, every single word is classified into one of the categories.However, in LSTM we can use a multiple word string to find out the class to which it belongs. This is very helpful while working with Natural language processing. If we use appropriate layers of embedding and encoding in LSTM, the model will be able to find out the actual meaning in input string and will give the most accurate output class. It can also mainly to solve the problems of gradient disappearance and gradient explosion in the process of long sequence training. In short, LSTM usually can perform better in longer sequences than ordinary RNN and other non-neural network classification techniques.  

In order to improve our accuracy, one hot is used to encode categorical variables as binary vectors. And set the Epoch 12, the bach size 256 (which is constricted by colab RAM room) After Lstm layer, a Fully-connected layer follows. Then the drop out layer is used to prevent the net from overfitting. In the end, the dense layer was conducted to print output.

<img width="500" height="200" src="https://github.com/zhanglu980608/NLP-Final-Project/blob/main/pics/lstm_layers.png">
Here are the evaluations for each epoch.
<img width="800" src="https://github.com/zhanglu980608/NLP-Final-Project/blob/main/pics/lstm_layers_evaluations.png">
The accuracy of train dataset and validation dataset are both increasing while the loss on train dataset is decreasing, but on validation dataset, the loss decreases first and increase a little later. Maybe there are some overfitting problems. However, in genral, the lstm model performs well. 
For test dataset, acc:0.71893, pre:0.71940, recall:0.71780, f1:0.71640. 

# 4. Model Evaluation
## 4.1 Model comparation
| Model         | Multinomial Navie Bayes | Perceptron | SVM | Decision Tree | Random Forest | BERT   |  LSTM   |
| ------------- | ----------------------- | ---------- | ----|-------------- | ------------- | -----  | ------  |
| Accuracy      |          0.84           |   0.81     |0.85 |     0.69      |    0.79       | 0.786  |  0.718  |
| Precision     |          0.84           |   0.81     |0.85 |     0.69      |    0.78       | 0.787  |  0.719  |
| Recall        |          0.84           |   0.81     |0.85 |     0.69      |    0.78       | 0.786  |  0.717  |
| F-1 Score     |          0.84           |   0.81     |0.85 |     0.69      |    0.78       | 0.786  |  0.716  |


## 4.2 Feature importance
![image](https://user-images.githubusercontent.com/89607189/144779460-6d7eaff9-5757-47e5-a126-790b13237718.png)


# 5. App Deployment
We used Streamlit, an open-source python framework, to deploy Web Apps for the models.
Our first Web App compares the Random Forest, Multinomial Naïve Bayes, and Support Vector Machine models for our test dataset. We can cross compare the predicted values as well as probability distributions.
Our second Web App enables users to input news headlines, and predicts the News headline classification and probability distributions using Multinomial Naïve Bayes model.

### Compare Models on Test Dataset:

![image](https://github.com/zhanglu980608/NLP-Final-Project/blob/main/pics/final_web.gif)

### User Input News Headlines:

![image](https://github.com/zhanglu980608/NLP-Final-Project/blob/main/pics/final_web_input_news_headline.gif)

