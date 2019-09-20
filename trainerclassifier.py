import pickle  # For Saving Model
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition, ensemble
import pandas


def train_model(valid_y,classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    saved_model = pickle.dumps(classifier) 
    filename='model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    return metrics.accuracy_score(predictions, valid_y)
texts=[]
def model_builder():
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
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)   

    accuracy = train_model(valid_y,naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    
    return str(accuracy)

def add_data_and_retrain_model(data):
    reru=""
    for i in range(len(data)):
        reru = reru + data[i]+","
        with open("Dataset\\BadWords.txt", "a") as myfile:
            myfile.write(data[i]+"\n")
            myfile.close()
    acuuracy=model_builder()
    return reru
print(model_builder())