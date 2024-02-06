import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize  # found when researching not sure if this was required


"""
Reads csv file into dataframe.
Gets category variables.
Combines the two relevant features after stripping unnecessary characters.
"""
df = pd.read_csv("reduced_huffpost.csv", encoding="latin-1") #encoding ensures special characters are being read in correctly
labels = df["category"]
df["headline"] = df["headline"].str.strip("b\'\"")
df["short_description"] = df["short_description"].str.strip("b\'\"")
documents = df["headline"] + df["short_description"]

"""
Perform labelencoding.
"""
le = LabelEncoder()
categoriesEncoded = le.fit_transform(labels)

"""
Apply the different types of stemming and lemmatization.
"""
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

porterStemmer = PorterStemmer()
snowballStemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def preProcesser(documents, technique):
    processed_docs = []
    for doc in documents:
        tokens = word_tokenize(doc)
        if technique == "porter":
            processed_docs.append(' '.join([porterStemmer.stem(token) for token in tokens]))
        elif technique == "snowball":
            processed_docs.append(' '.join([snowballStemmer.stem(token) for token in tokens]))
        elif technique == "lemmatizer":
            processed_docs.append(' '.join([lemmatizer.lemmatize(token) for token in tokens]))
        else:
            processed_docs.append(' '.join(tokens))  
    return processed_docs

"""
Using 10-fold cross validation evaluate the different vector representations.
Train and test using the Naive Bayes model.
Calculate the average accuracy and (micro and macro) F1-scores.
"""

def evalModel(preprocessing_technique, vectorization_technique):
    processed_docs = preProcesser(documents, preprocessing_technique)
    if vectorization_technique == "bow":
        vectorizer = CountVectorizer()
    elif vectorization_technique == "tfidf":
        vectorizer = TfidfVectorizer()
    else: 
        vectorizer = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=100), MinMaxScaler())#left minmax empty rather than copy= false 
    X = vectorizer.fit_transform(processed_docs)

    cv = KFold(n_splits=10)
    accuracies = []
    f1_micros = []
    f1_macros = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = categoriesEncoded[train_index], categoriesEncoded[test_index]
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_micros.append(f1_score(y_test, y_pred, average='micro'))
        f1_macros.append(f1_score(y_test, y_pred, average='macro'))
    return np.mean(accuracies), np.mean(f1_micros), np.mean(f1_macros)


techniques = ["none", "porter", "snowball", "lemmatizer"]
vectorizations = ["bow", "tfidf", "lsa"]

for preprocessing in techniques:
    for vectorization in vectorizations:
        accuracy, f1_micro, f1_macro = evalModel(preprocessing, vectorization)
        print(f"Preprocessing: {preprocessing}, Vectorization: {vectorization}, Accuracy: {accuracy}, F1 Micro: {f1_micro}, F1 Macro: {f1_macro}")
        
        
#data for answering part 2
print("Top Five Categories and their Counts:")
print(df['category'].value_counts().head(5))
category_counts = df['category'].value_counts()
print("Is dataset balanced or imbalanced? = ")
if min(category_counts) / max(category_counts) > 0.5:
    print("Balanced")
else:
    print("Imbalanced")
