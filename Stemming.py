import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
paragraph = """Excuses sound best, to the person who’s making them up. But, life has no place for excuses. 
Life only has only a little time. All though it will continue to go on, you cannot hold on. And think that every day is promised to you. 
A promise to live your dream! No one knows what that dream you have.
 No one cares how disappointing it might have been as you’ve been working toward that dream. 
 But that dream that you’re holding in your mind, is possible. When you run towards your dream, 
 life has a special kind of meaning. A meaning that can give birth to a great person like, late Dr. APJ Abdul Kalam did.
 He did recognize himself at a very early age. He became one of the best scientists and presidents India has ever had."""
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()
#Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ''.join(words)
    """
