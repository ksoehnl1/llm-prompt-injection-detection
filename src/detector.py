import pickle
import numpy as np
import re
import string
import os

class MaliciousPromptDetector:

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found at: {os.path.abspath(self.model_dir)}")

        self.vectorizer = None
        self.svm_model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        print("Loading models into memory...")

        with open(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)

        with open(os.path.join(self.model_dir, 'svm_classifier.pkl'), 'rb') as f:
            self.svm_model = pickle.load(f)

        with open(os.path.join(self.model_dir, 'feature_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        print("All models loaded successfully.")

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
        text = re.sub(r'[^\w\s.,!?;:()"\'-]', ' ', text)
        return text.strip()

    def extract_features(self, texts):
        features = []

        jailbreak_patterns = [
            'act as', 'pretend to be', 'roleplay', 'you are now', 'simulate',
            'persona', 'character', 'role', 'play the role', 'assume the role',
            'ignore previous', 'forget instructions', 'disregard', 'override',
            'bypass', 'jailbreak', 'unrestricted', 'uncensored', 'unfiltered',
            'without restrictions', 'no limitations', 'no rules', 'ignore all',
            'forget all', 'disregard all', 'override all', 'system prompt',
            'developer mode', 'admin mode', 'debug mode', 'dan mode'
        ]
        combination_patterns = [
            ['ignore', 'previous'], ['forget', 'instructions'], ['bypass', 'safety'],
            ['developer', 'mode'], ['admin', 'mode'], ['act', 'as']
        ]

        for text in texts:
            # Basic length features
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = np.mean([len(word) for word in text.split()]) if word_count > 0 else 0

            # Repetition features
            words = text.split()
            unique_words = len(set(words))
            repetition_ratio = unique_words / word_count if word_count > 0 else 0

            # Punctuation analysis
            punct_count = sum(1 for char in text if char in string.punctuation)
            punct_ratio = punct_count / char_count if char_count > 0 else 0

            # Capital letters ratio
            caps_count = sum(1 for char in text if char.isupper())
            caps_ratio = caps_count / char_count if char_count > 0 else 0

            # Suspicious patterns
            pattern_count = sum(1 for pattern in jailbreak_patterns if pattern in text.lower())

            # Combination pattern detection
            combination_score = 0
            for combo in combination_patterns:
                words_in_text = text.lower().split()
                if all(word in words_in_text for word in combo):
                    combination_score += 1

            authority_count = sum(1 for word in ['permission', 'authorized'] if word in text.lower())
            negation_count = sum(1 for word in ['not', 'no', 'never', 'without', 'ignore'] if word in text.lower())
            imperative_count = sum(1 for word in ['act', 'pretend', 'ignore', 'forget'] if word in text.lower())
            tech_count = sum(1 for word in ['system', 'mode', 'protocol', 'debug'] if word in text.lower())
            sentence_count = max(1, len([s for s in text.split('.') if s.strip()]))
            avg_sentence_length = word_count / sentence_count
            question_count = text.count('?')
            exclamation_count = text.count('!')
            quote_count = text.count('"') + text.count("'")
            paren_count = text.count('(') + text.count('[') + text.count('{')
            word_diversity = len(set(words)) / len(words) if words else 1.0

            features.append([
                char_count, word_count, avg_word_length, repetition_ratio,
                punct_ratio, caps_ratio, pattern_count, combination_score,
                authority_count, negation_count, imperative_count, tech_count,
                avg_sentence_length, question_count, exclamation_count,
                quote_count, paren_count, word_diversity
            ])

        return np.array(features)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        processed_texts = [self.preprocess_text(text) for text in texts]
        tfidf_features = self.vectorizer.transform(processed_texts)
        additional_features = self.extract_features(processed_texts)

        combined_features = np.hstack([tfidf_features.toarray(), additional_features])
        scaled_features = self.scaler.transform(combined_features)

        predictions = self.svm_model.predict(scaled_features)
        probabilities = self.svm_model.predict_proba(scaled_features)

        results = []
        for i, pred in enumerate(predictions):
            label = "Malicious" if pred == 1 else "Benign"
            confidence = probabilities[i].max()
            results.append({"prediction": label, "confidence": float(confidence)})

        return results