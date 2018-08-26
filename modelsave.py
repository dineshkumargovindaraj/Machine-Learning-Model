# Assignment 1 from Dinesh Kumar Govindaraj - A00421724
# Inspired from https://nlpforhackers.io/text-classification/ tried the same for effective learning.

from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle
from nltk.stem import PorterStemmer
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')

news = fetch_20newsgroups(subset='all')
print (len(news.data))
print (len(news.target_names))
print (news.target_names)

for text, num_label in zip(news.data[:10], news.target[:10]):
    print ('[%s]:\t\t "%s ..."' % (news.target_names[num_label], text[:100].split('\n')[0]))

    
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    classifier.fit(X_train, y_train)
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

model = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('classifier', MultinomialNB(alpha=0.05)),
])
 
train(model, news.data, news.target)

# save the model to disk
filename = 'trained_model.sav'
pickle.dump(model, open("C:\\Users\\dkdin\\Desktop\\DataMining\\DataMining\\Assignment1\\" + filename, 'wb'))