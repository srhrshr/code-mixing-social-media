from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer,confusion_matrix,f1_score,precision_recall_fscore_support
from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.externals import joblib
import datetime, time
import warnings
warnings.filterwarnings("ignore")

class CodeMixing():
	def __init__(self, data_path):
		self.data_path = data_path
		print "data_path:%s" %(self.data_path)
		self.load_data()
		self.train()
		self.save_and_report()
		# self.test()
		# X_test = [sent2features(s) for s in test_sents]
		# y_test = [sent2labels(s) for s in test_sents]
	
	def load_data(self):
		self.data = []
		with open(self.data_path) as ip:
			sent = []
			for line in ip.readlines():
				line = line.strip()
				if line != "":
					sent.append(tuple(line.split("\t")))
				else:
					self.data.append(sent)
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
		X_train = [self.sent2features(s) for s in self.data[0:int(0.8*len(self.data))]]
		y_train = [self.sent2labels(s) for s in self.data[0:int(0.8*len(self.data))]]

		X_test = [self.sent2features(s) for s in self.data[int(0.8*len(self.data)):]]
		y_test = [self.sent2labels(s) for s in self.data[int(0.8*len(self.data)):]]

		# crf = sklearn_crfsuite.CRF(
		# 	algorithm='lbfgs',
		# 	c1=0.1,
		# 	c2=0.1,
		# 	max_iterations=100,
		# 	all_possible_transitions=True)

		# crf.fit(X_train, y_train)

		# define fixed parameters and parameters to search
		crf = sklearn_crfsuite.CRF(
		    algorithm='lbfgs',
		    max_iterations=100,
		    all_possible_transitions=True
		)
		# labels = list(crf.classes_)
		params_space = {
		    'c1': scipy.stats.expon(scale=0.5),
		    'c2': scipy.stats.expon(scale=0.05),
		}

		# use the same metric for evaluation
		f1_scorer = make_scorer(metrics.flat_f1_score,
		                        average='weighted')

		# search
		rs = RandomizedSearchCV(crf, params_space,
		                        cv=10,
		                        verbose=1,
		                        n_jobs=-1,
		                        n_iter=50,
		                        scoring=f1_scorer)
		rs.fit(X_train, y_train)
		self.best_params = rs.best_params_
		self.best_cv_score = rs.best_score_
		print('best params:', rs.best_params_)
		print('best CV score:', rs.best_score_)
		# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

		crf = rs.best_estimator_
		
		y_pred = crf.predict(X_test)
		# print labels
		y_pred_flat = [item for sublist in y_pred for item in sublist]
		y_test_flat = [item for sublist in y_test for item in sublist]		
		
		print(confusion_matrix(y_test_flat, y_pred_flat))
		precision, recall, f1_score, support = precision_recall_fscore_support(y_test_flat, y_pred_flat,average='weighted')
		print "precision: %f, recall: %f, f1-score: %f, support: %s" %(precision, recall, f1_score, support)
		
		self.clf = crf
		self.accuracy = {"precision": precision, "recall": recall, "f1_score":f1_score }

	def save_and_report(self):
		report_file_name = datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m_%Y_%H_%M_%S")
		with open("experiments/reports/"+report_file_name,"w") as op:
			op.write("best cv score: %f" %(self.best_cv_score)+"\n")
			op.write("best params: %s" %(self.best_params)+"\n")
			op.write("data_path:%s" %(self.data_path)+"\n")
			op.write("accuracy: " + str(self.accuracy)+"\n")
		joblib.dump(self.clf, "experiments/models/"+report_file_name+".pkl")

if __name__ == '__main__':

	CodeMixing("data/Data-2016/Coarse-Grained/WA_TE_EN_CR.txt")
	# CodeMixing("data/Data-2016/Coarse-Grained/WA_HI_EN_CR.txt")