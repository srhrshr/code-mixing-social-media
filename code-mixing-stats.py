import numpy as np
class CodeMixingStats(object):
	"""docstring for CodeMixingStats"""
	def __init__(self, data_path1,data_path2,lang_code1,lang_code2,lang_code3,lang_code4):
		self.data_path1 = data_path1
		self.data_path2 = data_path2
		self.lang_code1 = lang_code1
		self.lang_code2 = lang_code2
		self.lang_code3 = lang_code3
		self.lang_code4 = lang_code4
		self.load_data()
		self.get_cmi_stats()
		print "num_utt:%d cmi_mixed:%f cmi_all:%f cmi_fraction:%f" %(self.num_utt,self.cmi_mixed,self.cmi_all,self.cmi_fraction)
		print "%d	%f	%f	%f" %(self.num_utt,self.cmi_mixed,self.cmi_all,self.cmi_fraction)
		# print self.cmi_mixed
		# print self.cmi_all
		# print self.cmi_fraction

	def load_data(self):
		self.data = []
		with open(self.data_path1) as ip1, open(self.data_path2) as ip2:
			sent = []
			for line in ip1.readlines() + ip2.readlines():
				line = line.strip()
				if line != "":
					sent.append(tuple(line.split("\t")))
				else:
					self.data.append(sent)
					sent = []
	def get_num_utterances(self):
		return len(self.data)

	def get_cmi_utt(self,sent,lang1,lang2,lang3,lang4):
		num_lang_1 = 0
		num_lang_2 = 0
		num_lang_3 = 0
		num_lang_4 = 0
		num_other = 0
		num_all = len(sent)
		for token_lang_pos in sent:
			try:
				token,lang,pos = token_lang_pos
			except:
				try:
					token,lang = token_lang_pos
				except:
					continue
			if lang == lang1:
				num_lang_1+=1
			elif lang == lang2:
				num_lang_2+=1
			elif lang == lang3:
				num_lang_3+=1
			elif lang == lang4:
				num_lang_4+=1
			else:
				num_other+= 1
		CMI = 0	
		if num_all > num_other:
			CMI = 100 * (1 - float(max(num_lang_1,num_lang_2,num_lang_3,num_lang_4))/float((num_all-num_other)))
		return CMI

	def get_cmi_stats(self):
		self.num_utt = len(self.data)
		sent_cmi_list = []
		for sent in self.data:
			sent_cmi_list.append(self.get_cmi_utt(sent,self.lang_code1,self.lang_code2,self.lang_code3,self.lang_code4))
		self.cmi_mixed = np.mean([x for x in sent_cmi_list if x!= 0.0])
		self.cmi_all = np.mean(sent_cmi_list)
		self.cmi_fraction = 100 * float(len([x for x in sent_cmi_list if x!= 0.0]))/float(len(sent_cmi_list))

import os		
if __name__ == '__main__':
	CodeMixingStats("data/Data-2016/Test/HI_Test/FB_EN_HI_Test_raw.txt","data/Data-2016/Coarse-Grained/FB_HI_EN_CR.txt","en","te","hi","bn")
