# News Headline Classification 
Team members: Keqin Liu, Linpei Zhang, Lu Zhang
## 1.Introduction
NLP applications are very popular in text classification and sentiment analysis. In this project, we want to 
## 2.Data Preparation
### 2.1 Data Collection
Data is collected from .... with labels. The dataset is split into 2 parts, 80% training dataset and 20% testing dataset. 
### 2.2 Text Cleaning & Tokenization
### 2.3 Feature Extraction: TF-IDF
## 3.Model
The models we leverage can be divided into two types: one is non-neural network and the other is neural network, which is LSTM. Meanwhile, non-neural networks consists of single machine learning models, which includes SVM, Decision Tree, and Multinomial Navie Bayes, and ensemble learning models, which includes random forest based on bagging and GBDT based on boosting.
<img width="1107" alt="models" src="https://user-images.githubusercontent.com/89560257/139080695-57b6ce7c-2a54-48a5-ae6a-a96142c78ae7.png">
### 3.1 SVM
Support Vector Machine is a supervised machine learning algorithm,which is mostly used in classification problems. In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.
### 3.2 Multinomial Navie Bayes
Multinomial Naive Bayes algorithm is based on the Bayes theorem. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output. The principle of it is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature.
### 3.3 Decision Tree
A decision tree is a prediction model structured in the form of a tree that is built using recursive splitting of the internal nodes which represents testing on certain features. The branches of the tree represent the outcome of the test and each leaf node denotes the final outputs that are labels in our case. 
It follows a top-down where the most important feature is located at the top (known as the root node) and the leaves represent corresponding classes/labels. Here, we will be using it to perform the classification of the news headlines.
### 3.4 Random Forest
A random forest is an ensemble classifier that estimates based on the combination of different decision trees. Effectively, it fits a number of decision tree classifiers on various subsamples of the dataset. Also, each tree in the forest built on a random best subset of features. Finally, the act of enabling these trees gives us the best subset of features among all the random subsets of features. 
### 3.5 GBDT
In gradient boosting decision trees, we combine many weak learners to come up with one strong learner. The weak learners here are the individual decision trees.
All the trees are conncted in series and each tree tries to minimise the error of the previous tree. Due to this sequential connection, boosting algorithms are usually slow to learn, but also highly accurate. In statistical learning, models that learn slowly perform better.
### 3.6 LSTM
Long short term memory is a special RNN model, which is effective in memorizing important information. We could find that other non-neural network classification techniques they are trained on multiple word as separate inputs that are just word having no actual meaning as a sentence, and while predicting the class it will give the output according to statistics and not according to meaning. That means, every single word is classified into one of the categories.HOwever, in LSTM we can use a multiple word string to find out the class to which it belongs. This is very helpful while working with Natural language processing. If we use appropriate layers of embedding and encoding in LSTM, the model will be able to find out the actual meaning in input string and will give the most accurate output class. It can also mainly to solve the problems of gradient disappearance and gradient explosion in the process of long sequence training. In short, LSTM usually can perform better in longer sequences than ordinary RNN and other non-neural network classification techniques.
## 4.Model Evaluation
Accuracy, precision, recall and F1 score are used to evaluate the performance of four models.\
Accuracy = correct predictions/all predictions\
Precision = true positive/(true positive + false positive)\
Recall = true positive/(true positive +false negative)\
F1 score = 2 precision * recall/(precision + recall)

## 5.Biggest Challenges
1. In previous research, the multinomial naive bayes is proved to perform best among all of the models. We want to verify this conclusion in our project.


