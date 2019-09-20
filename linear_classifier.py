from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
def train_model(valid_y,classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

def get_accuracy2():
    labels, texts = [],[]
    #reading good and Bad dataset file
    with open("Dataset//BadWords.txt") as fp:
        data=fp.readlines()
        for abc in data:
            labels.append("0")
            texts.append(abc)
    with open("Dataset//Goodwords.txt") as fp:
        data=fp.readlines()
        for abc in data:
            labels.append("1")
            texts.append(abc)
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    trainDF['label'] = labels

    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y) 

    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)     
    
    # Naive Bayes on Count Vectors
    accuracy = train_model(valid_y,linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    
    stri =""
    stri=stri+ "LC, Count Vectors: "+str(accuracy)+" + "

# Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(valid_y,linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    stri=stri+"LC, WordLevel TF-IDF: "+str(accuracy)+" + "

# Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(valid_y,linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    stri=stri+"LC, N-Gram Vectors: "+str(accuracy)+" + "

# Naive Bayes on Character Level TF IDF Vectors 
    accuracy = train_model(valid_y,linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    stri=stri+"LC, CharLevel Vectors: "+str(accuracy)+" + "
    return stri



