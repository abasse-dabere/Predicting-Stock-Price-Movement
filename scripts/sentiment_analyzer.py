import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        """Initialize the sentiment analyzer with the specified model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove retweets (RT) - ensure RT is a standalone word
        text = re.sub(r'\bRT\b\s+', '', text)
        # Remove mentions (e.g., @username)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        # Remove the '#' symbol from hashtags
        text = re.sub(r'#', '', text)
        # Remove hyperlinks (http or https)
        text = re.sub(r'https?:\/\/\S+', '', text)
        # Remove ':' symbol
        text = re.sub(r':', '', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        # Split text into individual words
        words = text.split()
        # Remove stop words
        words = [word for word in words if word not in self.stop_words]
        # Join words back together
        return ' '.join(words)

    def calculate_sentiment_score(self, text):
        """Calculate the sentiment score for a given text."""
        text = self.preprocess_text(text)
        max_tokens = self.tokenizer.model_max_length
        tokens = self.tokenizer.tokenize(text)

        # Split the text into chunks to respect the token limit
        chunks = [" ".join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

        scores = {"positive": 0, "negative": 0, "neutral": 0}
        total_chunks = len(chunks)

        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_tokens)
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Add the chunk scores
            scores["positive"] += probabilities[0][0].item()
            scores["negative"] += probabilities[0][1].item()
            scores["neutral"] += probabilities[0][2].item()

        if total_chunks == 0:
            return {"positive": 0, "negative": 0, "neutral": 0}

        # Normalize the scores by the number of chunks
        for key in scores:
            scores[key] /= total_chunks

        return scores

    def analyze_sentiments(self, texts, mode='sep'):
        """Analyze the sentiments of a list of texts.

        Args:
            texts (list): List of texts to analyze.
            mode (str): Analysis mode, 'sep' for separate scores, 'agg' for aggregated value.

        Returns:
            list: if mode='sep', a list of tuples with (negative, neutral, positive) scores.
            list: if mode='agg', a list of aggregated sentiment scores between -1 (negative) and 1 (positive).
        """
        results = []

        for text in texts:
            scores = self.calculate_sentiment_score(text)
            neg, neut, pos = scores['negative'], scores['neutral'], scores['positive']

            if mode == 'sep':
                results.append((neg, neut, pos))
            elif mode == 'agg':
                aggregated_score = (-neg + pos) / (neg + neut + pos)
                results.append(aggregated_score)
            else:
                raise ValueError("Invalid mode. Choose 'sep' or 'agg'.")

        return results