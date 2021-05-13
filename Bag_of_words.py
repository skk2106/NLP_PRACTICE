#Bag of words
import nltk
paragraph = """A spam filter is an application of sentence classification where it receives an email message and assigns whether it’s a spam or not. 
            If you want to classify news articles into different topics (business, politics, sports, etc.), it’s also a sentence classification task. 
            Sentence classification is one of the simplest NLP tasks that have a wide range of applications including document classification, spam filtering, and sentiment analysis. 
            Specifically, we’re going to look at the sentiment classifier and discuss its components in detail."""


#Cleaning the text i.e. to lower the sentences
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
        
    
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
