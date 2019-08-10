import nltk
import contractions
from sklearn.feature_extraction import text
import re

'''
This class does the pre-processing operations like tokenize, normalize and stem/ lemmatize
done on the fake news dataset and returns the preprocessed clean data set.

Will add comments later
 
Input: 

Output: 
'''


class Preprocess:

    def __init__(self, headline, body, preprocess_type):
        self.headline = headline
        self.body = body
        self.preprocess_type = preprocess_type

    # Replace contractions in the text
    def text_contractions(self, text):
        return contractions.fix(text)

    # Tokenizing the text
    def text_tokenize(self, text):
        return self.stopwords_removal(nltk.word_tokenize(text))

    # Remove punctuations and alphanumerics from text
    def remove_punctuation_alphanumerics_and_lowercase(self, text):
        punc_removal = " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
        return "".join(i for i in punc_removal if not i.isdigit())

    # Remove stop words from tokenized words
    def stopwords_removal(self, words):
        modified_words = []
        for word in words:
            if word not in text.ENGLISH_STOP_WORDS:
                modified_words.append(word)
        return modified_words

    # Stem words in tokenized words
    def stem_words(self, words, stemmer):
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    # Lemmatize words in tokenized words
    def lemmatize_words(self, words, lemmatizer):
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word=word)
            lemmas.append(lemma)
        return lemmas

    def stem_or_lemmatize(self, words):
        if self.preprocess_type == "stem":
            stemmer = nltk.stem.porter.PorterStemmer()
            return self.stem_words(words, stemmer)
        elif self.preprocess_type == "lemma":
            lemmatizer = nltk.WordNetLemmatizer()
            words = self.lemmatize_words(words, lemmatizer)
            return words
        else:
            raise Exception("Wrong pre-processing type chosen to clean the test. Try again!")

    def get_clean_headlines_and_bodies(self):
        bodies = []
        headlines = []

        def clean(text_data):
            text_data = self.text_contractions(text_data)
            text_data = self.remove_punctuation_alphanumerics_and_lowercase(text_data)

            words = self.text_tokenize(text_data)
            words = self.stopwords_removal(words)
            words = self.stem_or_lemmatize(words)

            return words

        for i in range(len(self.headline)):
            bodies.append(clean(self.body[int(self.headline[i]['Body ID'])]))
            headlines.append(clean(self.headline[i]['Headline']))

        return headlines, bodies
