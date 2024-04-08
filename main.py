import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import text_to_word_sequence


nltk.download('stopwords')

Sentence = '''It's important to note that "The Rudest Book Ever" has generated controversy for its blunt language and unconventional advice. While some readers appreciate its straightforward approach, others may find it offensive or inappropriate. As with any self-help book, it's essential to approach the content critically and consider how it aligns with your values and beliefs.'''

tokens= word_tokenize(Sentence)

token_words = [word for word in tokens if word.isalnum() and word.lower() not in stopwords.words('english')]

clean_sentence = ' '.join(token_words)
