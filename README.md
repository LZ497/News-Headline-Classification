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
### 3.1 SVM
### 3.2 Logistic Regression
### 3.3 Decision Tree
A decision tree is a prediction model structured in the form of a tree that is built using recursive splitting of the internal nodes which represents testing on certain features. The branches of the tree represent the outcome of the test and each leaf node denotes the final outputs that are labels in our case. 
It follows a top-down where the most important feature is located at the top (known as the root node) and the leaves represent corresponding classes/labels. Here, we will be using it to perform the classification of the news headlines.
### 3.4 Random Forest
A random forest is an ensemble classifier that estimates based on the combination of different decision trees. Effectively, it fits a number of decision tree classifiers on various subsamples of the dataset. Also, each tree in the forest built on a random best subset of features. Finally, the act of enabling these trees gives us the best subset of features among all the random subsets of features. 
### 3.5 Multinomial Navie Bayes
Multinomial Naive Bayes algorithm is based on the Bayes theorem. It calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output. The principle of it is each feature being classified is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of the other feature.
## 4.Model Evaluation
Accuracy, precision, recall and F1 score are used to evaluate the performance of four models.\
Accuracy = correct predictions/all predictions\
Precision = true positive/(true positive + false positive)\
Recall = true positive/(true positive +false negative)\
F1 score = 2 precision * recall/(precision + recall)

## 5.Biggest Challenges



