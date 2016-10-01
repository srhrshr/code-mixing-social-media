from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


class CodeMixing():
	def __init__(self, data_path):
		self.data_path = data_path
		self.load_data()
		self.train()
		# self.test()
		# X_test = [sent2features(s) for s in test_sents]
		# y_test = [sent2labels(s) for s in test_sents]
	
	def load_data(self):
		self.train_sents = []
		with open(self.data_path) as ip:
			sent = []
			for line in ip.readlines():
				line = line.strip()
				if line != "":
					sent.append(tuple(line.split("\t")))
				else:
					self.train_sents.append(sent)
					sent = []
	
	def word2features(self, sent, i):
	    word = sent[i][0]
	    postag = sent[i][1]

	    features = {
	        'bias': 1.0,
	        'word.lower()': word.lower(),
	        'word[-3:]': word[-3:],
	        'word[-2:]': word[-2:],
	        'word.isupper()': word.isupper(),
	        'word.istitle()': word.istitle(),
	        'word.isdigit()': word.isdigit(),
	        'postag': postag,
	        'postag[:2]': postag[:2],
	    }
	    if i > 0:
	        word1 = sent[i-1][0]
	        postag1 = sent[i-1][1]
	        features.update({
	            '-1:word.lower()': word1.lower(),
	            '-1:word.istitle()': word1.istitle(),
	            '-1:word.isupper()': word1.isupper(),
	            '-1:postag': postag1,
	            '-1:postag[:2]': postag1[:2],
	        })
	    else:
	        features['BOS'] = True

	    if i < len(sent)-1:
	        word1 = sent[i+1][0]
	        postag1 = sent[i+1][1]
	        features.update({
	            '+1:word.lower()': word1.lower(),
	            '+1:word.istitle()': word1.istitle(),
	            '+1:word.isupper()': word1.isupper(),
	            '+1:postag': postag1,
	            '+1:postag[:2]': postag1[:2],
	        })
	    else:
	        features['EOS'] = True

	    return features


	def sent2features(self,sent):
	    return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self,sent):
	    return [label for token, postag, label in sent]

	def sent2tokens(self,sent):
	    return [token for token, postag, label in sent]

	def train(self):
		X_train = [self.sent2features(s) for s in self.train_sents]
		y_train = [self.sent2labels(s) for s in self.train_sents]
		crf = sklearn_crfsuite.CRF(
			algorithm='lbfgs',
			c1=0.1,
			c2=0.1,
			max_iterations=100,
			all_possible_transitions=True)

		crf.fit(X_train, y_train)
		labels = list(crf.classes_)
		# y_pred = crf.predict(X_test)
		# metrics.flat_f1_score(y_test, y_pred,
  #                     average='weighted', labels=labels)

if __name__ == '__main__':
	CodeMixing("data/Data-2016/Coarse-Grained/WA_TE_EN_CR.txt")