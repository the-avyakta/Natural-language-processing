import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import text_to_word_sequence


nltk.download('wordnet')

Sentence = '''Natural Language Processing (NLP) is a field of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language. It encompasses a variety of tasks such as text analysis, machine translation, sentiment analysis, and speech recognition. NLP techniques enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. With the exponential growth of textual data on the internet, NLP has become increasingly important in applications ranging from chatbots and virtual assistants to content recommendation systems and information extraction. Despite its challenges, NLP continues to advance rapidly, driven by innovations in deep learning, neural networks, and computational linguistics'''
tokens= word_tokenize(Sentence)

token_words = [word for word in tokens if word.isalnum() and word.lower() not in stopwords.words('english')]

clean_sentence = ' '.join(token_words)
stemmer = PorterStemmer()
stem_words= [stemmer.stem(word) for word in tokens]

lemmatizer = WordNetLemmatizer()
lemma_words= [lemmatizer.lemmatize(word) for word in tokens]
print('\n'*5)
print("stem words", stem_words)
print('\n'*5)

print(" lemma_words", lemma_words)
