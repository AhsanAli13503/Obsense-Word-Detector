import re  #for Removing number from text
import nltk
from nltk.corpus import stopwords #for removing most common used words from text

class Data:
    orignalText=""
    preprocesstext=""
    def __init__(self,orignalText):
        self.orignalText =orignalText
        self.preprocesstext=""
        
    def removing_numbers(self):
        self.preprocesstext=re.sub(r'\d+', '', self.preprocesstext)
    
    def removing_commonword(self):
        stop_words=stopwords.words('english')
        tokenized_words = self.preprocesstext.split()
        self.preprocesstext=[word for word in tokenized_words if word not in stop_words]
        
    def convert_lower_to_upper(self):
        return self.orignalText.lower()
 



