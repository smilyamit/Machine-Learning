
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting =3)

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')  #stopword contain list of words
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-z]',' ', dataset['Review'][i] )
    review = review.lower()
    review = review.split()  # split will change the string character into list of words
    ps = PorterStemmer()  
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # set used for large data
    
    review = ' '.join(review)
    corpus.append(review)

#Note PorterStemmer class have stem tool to simlify the english word
# Creating the bag of word models
from sklearn.feature_extraction.text import CountVectorizer # it convert text to matrix
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting the Naive Base to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






