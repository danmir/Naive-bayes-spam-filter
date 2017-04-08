from collections import defaultdict
from decimal import Decimal
from math import log
from operator import itemgetter
import re


def tokenize_text(text):
    text = list(re.findall(r"[\w']+", text))
    text = list(map(lambda w: w.lower(), text))
    return list(filter(lambda w: len(w) > 3, text))


class NaiveBayes:
    # https://habrahabr.ru/post/184574/
    def __init__(self, categories):
        self.words = defaultdict(dict)
        self.categories = {category: {'total': 0, 'word_count': 0}
                           for category in categories}
        self.training_examples = 0
        self.unique_words = set()

    def train(self, category, text):
        words = tokenize_text(text)
        for word in words:
            # Общее количество уникальных слов в словаре
            self.unique_words.add(word)
            self._inc_word_freq(word, category)

        # Количество текстов для данной категории для априорной вероятн. класса
        self.categories[category]['total'] += 1
        # Увелич колич обучающих слов для категории
        self._inc_category_word_count(category, len(words))
        # Количество тренировочных текстов
        self.training_examples += 1

    def _inc_word_freq(self, word, category):
        if self.words[word].get(category):
            self.words[word][category] += 1
        else:
            self.words[word][category] = 1

    def _inc_category_word_count(self, category, number):
        if self.categories[category].get('word_count'):
            self.categories[category]['word_count'] += number
        else:
            self.categories[category]['word_count'] = number

    def classify(self, text):
        words = tokenize_text(text)

        probabilities = {}
        for category, category_data in self.categories.items():
            category_prob = self._get_category_probability(category_data['total'])
            predictors_likelihood = self._get_predictors_probability(category, words)
            probabilities[category] = category_prob + predictors_likelihood

        return 1 if probabilities[1] > probabilities[0] else 0

    def _get_category_probability(self, count):
        # Class prior probability
        return 0.5
        #return log(Decimal(float(count)) / Decimal(self.training_examples + len(self.categories.keys())))

    def _get_predictors_probability(self, category, words):
        likelihood = 1
        for word in words:
            likelihood += log(self._P_feat_class(category, word))

        return likelihood

    def _P_feat_class(self, category, word):
        z = 1  # Laplace coeff
        word_count = self.categories[category]['word_count'] + z * len(self.unique_words)
        if not self.words.get(word) or not self.words[word].get(category):
            smoothed_freq = z  # Laplace smoothing
        else:
            smoothed_freq = z + self.words[word][category]
        return Decimal(float(smoothed_freq)) / Decimal(word_count)

    def _collect_word_stat(self):
        category_prob_spam = self._get_category_probability(self.categories[1]['total'])
        category_prob_ham = self._get_category_probability(self.categories[0]['total'])
        # Для всех слов
        for word in self.words.keys():
            # print(self._P_feat_class(1, word), Decimal(category_prob_spam))
            p1 = log(self._P_feat_class(1, word) * Decimal(category_prob_spam))
            p2 = log(self._P_feat_class(0, word) * Decimal(category_prob_ham))
            metric = max(Decimal(p1) / Decimal(p2), Decimal(p2) / Decimal(p1))
            self.words[word]['metric'] = metric

    def get_stat(self):
        self._collect_word_stat()

        stat_dict = {}
        for word in self.words.keys():
            stat_dict[word] = self.words[word]['metric']
        dc_sort = sorted(stat_dict.items(), key=itemgetter(1), reverse=True)
        return dc_sort
