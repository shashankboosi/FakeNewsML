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

        return weight_features

