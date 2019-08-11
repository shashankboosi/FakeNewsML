from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


# This function describes the feature transformation for the data
class Features:
    def __init__(self, pre_headline, pre_body, headline, body):
        self.preprocessed_headlines = pre_headline
        self.preprocessed_bodies = pre_body
        self.headline = headline
        self.body = body

    def sentence_weighting(self):
        weight_features = []
        for headline, body in zip(self.preprocessed_headlines, self.preprocessed_bodies):
            intersection_of_sentences = set(headline).intersection(body)
            union_of_sentences = set(headline).union(body)

            feature_division = len(intersection_of_sentences) / float(len(union_of_sentences))
            weight_features.append(feature_division)

        return sp.coo_matrix(weight_features)

    def cosine_sim(self, tfidf_weights):
        return sp.coo_matrix(cosine_similarity(tfidf_weights, tfidf_weights))

    def tfidf_extraction(self, validation_headlines, validation_bodies, test_headlines, test_bodies):
        tf_headline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True,
                                                 max_features=10000)
        tf_train_headline = tf_headline_vectorizer.fit_transform(self.headline)

        tf_body_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True,
                                             max_features=10000)
        tf_train_body = tf_body_vectorizer.fit_transform(self.body)

        tf_validation_headline = tf_headline_vectorizer.transform(validation_headlines)
        tf_validation_body = tf_body_vectorizer.transform(validation_bodies)

        tf_test_headline = tf_headline_vectorizer.transform(test_headlines)
        tf_test_body = tf_body_vectorizer.transform(test_bodies)

        train_tfidf = sp.hstack([tf_train_headline, tf_train_body])
        validation_tfidf = sp.hstack([tf_validation_headline, tf_validation_body])
        test_tfidf = sp.hstack([tf_test_headline, tf_test_body])

        return train_tfidf, validation_tfidf, test_tfidf

