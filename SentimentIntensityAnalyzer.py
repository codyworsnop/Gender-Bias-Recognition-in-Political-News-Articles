#######This file calculates sentiment (OLD) #######
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize 
import statistics
from sentiment import sentiment_analysis

class SentimentAnalyzer(): 

    def __init__(self): 
        self.Analyzer = SentimentIntensityAnalyzer()

    def AnalyzeSentiment(self, article: str, should_average_intensities=True):

        annotations = sentiment_analysis.analyze(article) 
        score = annotations.document_sentiment.score
        magnitude = annotations.document_sentiment.magnitude
        return score, magnitude
    