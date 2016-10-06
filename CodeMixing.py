from itertools import chain

import nltk
import re
import sklearn
import scipy.stats
import numpy as np
from sklearn.metrics import make_scorer,confusion_matrix,f1_score,precision_recall_fscore_support,average_precision_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import KFold
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
		#model_name = '06_10_2016_00_54_25'
		#self.test(model_name)
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
	    lang = sent[i][1]

	    word, has_emoji = self.remove_emoji(word)

	    #has http:// 
	    features = {
	        'bias': 1.0,
	        'word.lower()': word.lower(),
	        'word[0:2]': word[0:2],
	        'word[0:3]': word[0:3],
	        'word[-3:]': word[-3:],
	        'word[-2:]': word[-2:],
	        'word.isupper()': word.isupper(),
	        'word.istitle()': word.istitle(),
	        'word.isdigit()': word.isdigit(),
	        'word.has_emoji':has_emoji,
	        'word.has_num':self.has_num(word),
	        'lang': lang
	    }
	    self.add_char_ngram_features(word,[1,2,3],features,count='count')

	    if i > 0:
	        word1 = sent[i-1][0]
	        lang1 = sent[i-1][1]
	        word1, has_emoji = self.remove_emoji(word1)
	        features.update({
	            '-1:word.lower()': word1.lower(),
	            '-1:word.istitle()': word1.istitle(),
	            '-1:word.isupper()': word1.isupper(),
	            '-1:lang': lang1
	        })
	    else:
	        features['BOS'] = True

	    if i < len(sent)-1:
	        word1 = sent[i+1][0]
	        lang1 = sent[i+1][1]
	        word1, has_emoji = self.remove_emoji(word1)
	        features.update({
	            '+1:word.lower()': word1.lower(),
	            '+1:word.istitle()': word1.istitle(),
	            '+1:word.isupper()': word1.isupper(),
	            '+1:lang': lang1
	        })
	    else:
	        features['EOS'] = True

	    return features

	def add_char_ngram_features(self,word,n_list,features,count='binary'):
		for n in n_list:
			if not len(word) <= n:
				char_ngrams = self.word2ngrams(word,n)
				char_ngrams_set = list(set(char_ngrams))
				if count == 'binary':
					char_ngram_dict = {x:1 for x in char_ngrams_set}
				elif count == 'count':
					char_ngram_dict = {x:char_ngrams.count(x) for x in char_ngrams_set}
				features.update(char_ngram_dict)

	def word2ngrams(self,text, n=3):
		""" Convert word into character ngrams. """
		return [text[i:i+n] for i in range(len(text)-n+1)]
	
	def has_num(self,text):
		return bool(re.search(r'\d', text))

	def remove_emoji(self,text):
		try:
			text = text.decode('utf-8')
			emoji_pattern = re.compile("["
				u"\U0001F600-\U0001F64F"  # emoticons
				u"\U0001F300-\U0001F5FF"  # symbols & pictographs
				u"\U0001F680-\U0001F6FF"  # transport & map symbols
				u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
				"]+", flags=re.UNICODE)
			text_without_emoji = emoji_pattern.sub(r'', text)
			return text_without_emoji,len(text_without_emoji) != len(text)
		except:
			return text,False			

	def sent2features(self,sent):
	    return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self,sent):
	    return [label for token, postag, label in sent]

	def sent2tokens(self,sent):
	    return [token for token, postag, label in sent]

	def train(self):
		X = [self.sent2features(s) for s in self.data]
		y = [self.sent2labels(s) for s in self.data]
		
		# X_train = [self.sent2features(s) for s in self.data[0:int(0.8*len(self.data))]]
		# y_train = [self.sent2labels(s) for s in self.data[0:int(0.8*len(self.data))]]
		# # X_train = [self.sent2features(s) for s in self.data]
		# y_train = [self.sent2labels(s) for s in self.data]

		# X_test = [self.sent2features(s) for s in self.data[int(0.8*len(self.data)):]]
		# y_test = [self.sent2labels(s) for s in self.data[int(0.8*len(self.data)):]]

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
		_scorer = make_scorer(metrics.flat_accuracy_score)
		# _scorer = make_scorer(metrics.flat_f1_score,
		#                         average='weighted')

		# search
		rs = RandomizedSearchCV(crf, params_space,
		                        cv=4,
		                        verbose=1,
		                        n_jobs=-1,
		                        n_iter=50,
		                        scoring=_scorer)
		kf = KFold(n_splits=4)
		for train_index, test_index in kf.split(X):			
			X_train, X_test = self.get_subset(X,train_index), self.get_subset(X,test_index)
			y_train, y_test = self.get_subset(y,train_index), self.get_subset(y,test_index)
			rs.fit(X_train, y_train)
			self.best_params = rs.best_params_
			self.best_cv_score = rs.best_score_
			print('best params:', rs.best_params_)
			print('best CV score:', rs.best_score_)
			# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

			crf = rs.best_estimator_
			
			y_pred = crf.predict(X_test)
			y_test = y_test

			print metrics.flat_classification_report(y_test,y_pred)
			# print labels
			y_pred_flat = [item for sublist in y_pred for item in sublist]
			y_test_flat = [item for sublist in y_test for item in sublist]		
			
			precision, recall, f1_score, support = precision_recall_fscore_support(y_test_flat,y_pred_flat,average='weighted')
			print "precision: %f, recall: %f, f1-score: %f, support: %s" %(precision, recall, f1_score, support)
			
			self.clf = crf
			self.accuracy = {"precision": precision, "recall": recall, "f1_score":f1_score }
			self.save_and_report()

	def save_and_report(self):
		report_file_name = datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m_%Y_%H_%M_%S")
		with open("experiments/reports/"+report_file_name,"w") as op:
			op.write("best cv score: %f" %(self.best_cv_score)+"\n")
			op.write("best params: %s" %(self.best_params)+"\n")
			op.write("data_path:%s" %(self.data_path)+"\n")
			op.write("accuracy: " + str(self.accuracy)+"\n")
		joblib.dump(self.clf, "experiments/models/"+report_file_name+".pkl")

	def test(self,model_name):
		X_train = [self.sent2features(s) for s in self.data]
		y_train = [self.sent2labels(s) for s in self.data]

		clf = joblib.load("experiments/models/"+model_name+".pkl")
		y_pred = clf.predict(X_train)
		with open("experiments/output/"+model_name+".tsv","w") as op:
			for seq_index,seq in enumerate(X_train):
				gold_seq = y_train[seq_index]
				pred_seq = y_pred[seq_index]
				for word_index,item in enumerate(seq):
					word_value = item['word.lower()']
					lang = item['lang']
					gold_label = gold_seq[word_index]
					pred_label = pred_seq[word_index]
					op.write("\t".join([word_value,lang,gold_label,pred_label])+"\n")
				op.write("\n")

	def get_subset(self,X,index):
		return [X[x] for x in index]

import os		
if __name__ == '__main__':
	CodeMixing("data/Data-2016/Coarse-Grained/WA_HI_EN_CR.txt")
	# for file_name in os.listdir("data/Data-2016/Fine-Grained/"):
	# # CodeMixing("data/Data-2016/Coarse-Grained/WA_TE_EN_CR.txt")
	# 	CodeMixing("data/Data-2016/Fine-Grained/"+file_name)	