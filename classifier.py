import pickle  # For Saving Model
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition, ensemble
import pandas

def load_classfication_pred_res(data):
    clas = []
    clas.append(data)
    filename='model.sav'
    classifier =pickle.load(open(filename, 'rb'))
    texts = []
    with open("Dataset//BadWords.txt") as fp:
        data=fp.readlines()
        for abc in data:
            texts.append(abc)
    with open("Dataset//Goodwords.txt") as fp:
        data=fp.readlines()
        for abc in data:
            texts.append(abc)
    trainDF = pandas.DataFrame()
    trainDF['text'] = texts
    
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    classificationtext =  tfidf_vect_ngram_chars.transform(clas) 
    predictions = classifier.predict(classificationtext)

    predictions = classifier.predict(classificationtext)
    return predictions[0]


    