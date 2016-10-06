from itertools import chain

import nltk
import re
import string
import pickle
import sklearn
import scipy.stats
import numpy as np
from sklearn.metrics import make_scorer,confusion_matrix,f1_score,precision_recall_fscore_support,average_precision_score
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.cross_validation import KFold
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.utils import flatten
from sklearn.externals import joblib
import datetime, time
import warnings,traceback
warnings.filterwarnings("ignore")

coarse_fine_mapping_dict = pickle.load(open("coarse_fine_mapping_dict.pkl"))

class CodeMixing():

	def __init__(self,data_path):
		self.data_path = data_path
		self.social_media_name = data_path.split("/")[-1].strip(".txt") 
		print "data_path:%s" %(self.data_path)
		self.load_data()
		# for i in [1,2,3]:
		self.train()
		# self.model_name = '_TWT_BN_EN_CR'
		# self.test()
		
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
	
	def word2features(self,sent,i,ignore=False):
		word = sent[i][0]
		try:
			lang = sent[i][1]
		except:
			lang = 'unk'
		word, has_emoji = self.remove_emoji(word)

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
			'word.startswith_arobase':word.startswith('@'),
			'word.startswith_hash':word.startswith('#'),
			'word.web_address':self.is_web_address(word),
			'word.is_punct':self.is_punct(word),
			'lang': lang
		}

		if not (features['word.startswith_arobase'] or features['word.startswith_hash'] or features['word.web_address'] or features['word.is_punct']):
			self.add_char_ngram_features(word,[1,2,3],features)

		# if self.grain == "fine" and not ignore:
		# 	if len(sent[i]) == 3:#train
		# 		pos_tag = sent[i][2].strip()
		# 		coarse_pos_tag = self.get_coarse_from_dict(pos_tag)
		# 	else:#test
		# 		coarse_pos_tag = self.get_coarse_from_model(sent,i)
		# 		# print word, coarse_pos_tag
		# 	features.update({'coarse_pos_tag':coarse_pos_tag})
		
		if i > 0:
			word1 = sent[i-1][0]
			try:
				lang1 = sent[i-1][1]
			except:
				lang1 = 'unk'			
			word1, has_emoji1 = self.remove_emoji(word1)
			features.update({
				'-1:word.lower()': word1.lower(),
				'-1:word[0:2]': word1[0:2],
				'-1:word[0:3]': word1[0:3],
				'-1:word[-3:]': word1[-3:],
				'-1:word[-2:]': word1[-2:],
				'-1:word.isupper()': word1.isupper(),
				'-1:word.istitle()': word1.istitle(),
				'-1:word.isdigit()': word1.isdigit(),
				'-1:word.has_emoji':has_emoji1,
				'-1:word.has_num':self.has_num(word1),
				'-1:word.startswith_arobase':word1.startswith('@'),
				'-1:word.startswith_hash':word1.startswith('#'),
				'-1:word.web_address':self.is_web_address(word1),
				'-1:word.is_punct':self.is_punct(word1),
				'-1:lang': lang1
			})
		else:
			features['BOS'] = True

		if i < len(sent)-1:
			word1 = sent[i+1][0]
			try:
				lang1 = sent[i+1][1]
			except:
				lang1 = 'unk'			
			word1, has_emoji1 = self.remove_emoji(word1)
			features.update({
				'+1:word.lower()': word1.lower(),
				'+1:word[0:2]': word1[0:2],
				'+1:word[0:3]': word1[0:3],
				'+1:word[-3:]': word1[-3:],
				'+1:word[-2:]': word1[-2:],
				'+1:word.isupper()': word1.isupper(),
				'+1:word.istitle()': word1.istitle(),
				'+1:word.isdigit()': word1.isdigit(),
				'+1:word.has_emoji':has_emoji1,
				'+1:word.has_num':self.has_num(word1),
				'+1:word.startswith_arobase':word1.startswith('@'),
				'+1:word.startswith_hash':word1.startswith('#'),
				'+1:word.web_address':self.is_web_address(word1),
				'+1:word.is_punct':self.is_punct(word1),
				'+1:lang': lang1
			})
		else:
			features['EOS'] = True

		if i > 1:
			word2 = sent[i-2][0]
			try:
				lang2 = sent[i-2][1]
			except:
				lang2 = 'unk'			
			word2, has_emoji2 = self.remove_emoji(word2)
			features.update({
				'-2:word.lower()': word2.lower(),
				'-2:word[0:2]': word2[0:2],
				'-2:word[0:3]': word2[0:3],
				'-2:word[-3:]': word2[-3:],
				'-2:word[-2:]': word2[-2:],
				'-2:word.isupper()': word2.isupper(),
				'-2:word.istitle()': word2.istitle(),
				'-2:word.isdigit()': word2.isdigit(),
				'-2:word.has_emoji':has_emoji2,
				'-2:word.has_num':self.has_num(word2),
				'-2:word.startswith_arobase':word2.startswith('@'),
				'-2:word.startswith_hash':word2.startswith('#'),
				'-2:word.web_address':self.is_web_address(word2),
				'-2:word.is_punct':self.is_punct(word2),
				'-2:lang': lang2
			})

		if i < len(sent)-2:
			word2 = sent[i+2][0]
			try:
				lang2 = sent[i+2][1]
			except:
				lang2 = 'unk'
			word2, has_emoji2 = self.remove_emoji(word2)
			features.update({
				'+2:word.lower()': word2.lower(),
				'+2:word[0:2]': word2[0:2],
				'+2:word[0:3]': word2[0:3],
				'+2:word[-3:]': word2[-3:],
				'+2:word[-2:]': word2[-2:],
				'+2:word.isupper()': word2.isupper(),
				'+2:word.istitle()': word2.istitle(),
				'+2:word.isdigit()': word2.isdigit(),
				'+2:word.has_emoji':has_emoji2,
				'+2:word.has_num':self.has_num(word2),
				'+2:word.startswith_arobase':word2.startswith('@'),
				'+2:word.startswith_hash':word2.startswith('#'),
				'+2:word.web_address':self.is_web_address(word2),
				'+2:word.is_punct':self.is_punct(word2),
				'+2:lang': lang2
			})
		
		return features

	def is_web_address(self,word):
		return word.startswith("http") or \
			'.com' in word or \
			'.me' in word or \
			('s/' in word and self.has_num(word))

	def is_punct(self, word):
		try:
			word_puncts_removed = str(word).translate(None, string.punctuation)
			return len(word_puncts_removed) == 0
		except:
			return False
	
	# def get_coarse_from_dict(self,label=None):
	# 	for key in coarse_fine_mapping_dict.keys():
	# 		if label in coarse_fine_mapping_dict[key]:
	# 			return key
	# 	return label

	# def get_coarse_from_model(self,sent,i):
	# 	clf = joblib.load("experiments/models/"+self.coarse_model_name+".pkl")
	# 	feature_vector = [self.word2features(sent,i,ignore=True) for i in range(len(sent))]
	# 	pred = clf.predict(feature_vector)
	# 	print sent[i][0],len(sent),len(feature_vector), len(pred)
	# 	return pred
	# 	# for key in coarse_fine_mapping_dict.keys():
	# 	# 	if label in coarse_fine_mapping_dict[key]:
	# 	# 		return key
	# 	# return label

	def add_char_ngram_features(self,word,n_list,features,count=None):
		for n in n_list:
			if not len(word) <= n:
				char_ngrams = self.word2ngrams(word,n)
				char_ngrams_set = list(set(char_ngrams))
				if count == 'binary':
					char_ngram_dict = {x:1 for x in char_ngrams_set}
				elif count == 'count':
					char_ngram_dict = {x:char_ngrams.count(x) for x in char_ngrams_set}
				elif count==None:
					char_ngram_dict = {'Char_'+str(n)+'_gram_Pos_'+str(i):n_gram for i,n_gram in enumerate(char_ngrams)}
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
		except Exception:
			traceback.print_exc()
			return text,False			

	def sent2features(self,sent):
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self,sent):
		return [label for token, postag, label in sent]

	def sent2tokens(self,sent):
		try:
			return [token for token, postag in sent]
		except:
			x = []
			for i,z in enumerate(sent):
				x.append(z[0])
			return x

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
			'c2': scipy.stats.expon(scale=0.05)
		}
		# params_space = {
		# 	'algorithm': ['lbfgs','l2sgd']
		# }

		# use the same metric for evaluation
		_scorer = make_scorer(metrics.flat_precision_score,average='weighted')
		# _scorer = make_scorer(metrics.flat_f1_score,
		#						 average='weighted')

		# search
		rs = RandomizedSearchCV(crf, params_space,
								cv=4,
								verbose=1,
								n_jobs=2,
								scoring=_scorer)
		rs.fit(X, y)
		self.grid_scores = rs.grid_scores_
		self.best_params = rs.best_params_
		self.best_cv_score = rs.best_score_
		print('best params:', self.best_params)
		print('best CV score:', self.best_cv_score)
		# print('Grid scores:',  self.grid_scores)
		# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
		self.clf = rs.best_estimator_
		y_pred = self.clf.predict(X)
		y_test = y

		# print metrics.flat_classification_report(y_test,y_pred)
		# # print labels
		y_pred = flatten(y_pred)
		y_test = flatten(y_test)
		
		precision, recall, f1_score, support = precision_recall_fscore_support(y_test,y_pred,average='weighted')
		print "precision: %f, recall: %f, f1-score: %f, support: %s" %(precision, recall, f1_score, support)
		
		self.accuracy = {"metric":"f1_score","cv_score":self.best_cv_score}
		self.save_and_report()

	def save_and_report(self):
		report_file_name = datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m_%Y_%H_%M_%S")
		with open("experiments/reports/"+report_file_name+"_"+self.social_media_name,"w") as op:
			op.write("best cv score: %f" %(self.best_cv_score)+"\n")
			op.write("best params: %s" %(self.best_params)+"\n")
			op.write("data_path:%s" %(self.data_path)+"\n")
			op.write("accuracy: " + str(self.accuracy)+"\n")
		joblib.dump(self.clf, "experiments/models/"+"_"+self.social_media_name+".pkl")

	def test(self):
		X = [self.sent2features(s) for s in self.data]
		tokens = [self.sent2tokens(s) for s in self.data]
		clf = joblib.load("experiments/models/"+self.model_name+".pkl")
		y_pred = clf.predict(X)
		# precision, recall, f1_score, support = precision_recall_fscore_support(flatten(y_test),flatten(y_pred),average='weighted')
		# print "precision: %f, recall: %f, f1-score: %f, support: %s" %(precision, recall, f1_score, support)
		# count = 0
		# total = len(flatten(y_pred))
		# for ix,y in enumerate(flatten(y_pred)):
		# 	if y != flatten(y_test)[ix]:
		# 		print y, flatten(y_test)[ix]
		# 		count+=1
		# print count, total

		with open("experiments/output/"+self.model_name+"_"+self.social_media_name+".tsv","w") as op:
			for seq_index,seq in enumerate(X):
				pred_seq = y_pred[seq_index]
				token_seq = tokens[seq_index]
				for word_index,item in enumerate(seq):
					word_value = token_seq[word_index]
					lang = item['lang']
					pred_label = pred_seq[word_index]
					op.write("\t".join([word_value,lang,pred_label])+"\n")
				op.write("\n")

	def get_subset(self,X,index):
		return [X[x] for x in index]

import os		
if __name__ == '__main__':
	# CodeMixing("data/Data-2016/Fine-Grained/FB_HI_EN_FN.txt")
	# CodeMixing("data/Data-2016/Fine-Grained/FB_BN_EN_FN.txt")
	#CodeMixing("data/Data-2016/Fine-Grained/TWT_BN_EN_FN.txt")
	# CodeMixing("data/Data-2016/Fine-Grained/WA_HI_EN_FN.txt",'train')
	# CodeMixing("data/Data-2016/Test/BN_Test/BN_Test/TWT_BN_EN_FN_Test_raw.txt")
	# for file_name in os.listdir("data/Data-2016/Fine-Grained/"):
	CodeMixing("data/Data-2016/Coarse-Grained/TWT_TE_EN_CR.txt")
	# 	CodeMixing("data/Data-2016/Fine-Grained/"+file_name)	
