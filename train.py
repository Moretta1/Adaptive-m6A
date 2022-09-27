#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
import os
import random

import torch
import torch.nn as nn ##########
import torch.utils.data as Data
from torch.autograd import Variable
from torch import optim
from sklearn.model_selection import train_test_split
import itertools

from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc,precision_recall_curve
# from torch.utils.tensorboard import SummaryWriter
import seaborn as sns

#----->>
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

#-------------------------------------->>>>

def load_data_bicoding(Path):
	data = np.loadtxt(Path,dtype=list)
	data_result = []
	for seq in data:
		seq = str(seq.strip('\n'))
		data_result.append(seq)
	return data_result 
	
	
def transform_token2index(sequences): 
	token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))
	print(token2index) 
	
	for i, seq in enumerate(sequences):
		sequences[i] = list(seq)
	
	token_list = list()
	max_len = 0
	for seq in sequences:
		seq_id = [token2index[residue] for residue in seq]
		token_list.append(seq_id)
		if len(seq) > max_len:
			max_len = len(seq)
	
	print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
	print('sequences_residue', sequences[0:5]) 
	print('token_list', token_list[0:5])
	return token_list, max_len

def make_data_with_unified_length(token_list,max_len):
	token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))
	data = []
	for i in range(len(token_list)):
		token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']] #前
		n_pad = max_len - len(token_list[i])
		token_list[i].extend([0] * n_pad) 
		data.append(token_list[i])
	
	print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
	print('max_len + 2', max_len)
	print('token_list + [pad]', token_list[0:5])
	
	return data


	
def load_train_val_bicoding(path_pos_data, path_neg_data):
	
	sequences_pos = load_data_bicoding(path_pos_data)
	sequences_neg = load_data_bicoding(path_neg_data)
	
	token_list_pos, max_len_pos = transform_token2index(sequences_pos) #打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg) #打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len = max(max_len_pos,max_len_neg)
	
	Positive_X = make_data_with_unified_length(token_list_pos,max_len)
	Negitive_X = make_data_with_unified_length(token_list_neg,max_len)
	
	data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
	
	np.random.seed(42)
	np.random.shuffle(data_train)

	X = np.array([_[:-1] for _ in data_train])
	y = np.array([_[-1] for _ in data_train])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/8, random_state=42)

	return X_train, y_train, X_test, y_test
	
def load_test_bicoding(path_pos_data,path_neg_data):
	
	sequences_pos = load_data_bicoding(path_pos_data)
	sequences_neg = load_data_bicoding(path_neg_data)
	
	token_list_pos, max_len_pos = transform_token2index(sequences_pos) #打印1次
	token_list_neg, max_len_neg = transform_token2index(sequences_neg) #打印1次
	# token_list_train: [[1, 5, 8], [2, 7, 9], ...]
	max_len = max(max_len_pos,max_len_neg)
	
	Positive_X = make_data_with_unified_length(token_list_pos,max_len)
	Negitive_X = make_data_with_unified_length(token_list_neg,max_len)

	data_train = np.array([_ + [1] for _ in Positive_X] + [_ + [0] for _ in Negitive_X])
		
	np.random.seed(42)
	np.random.shuffle(data_train)

	X_test = np.array([_[:-1] for _ in data_train])
	y_test = np.array([_[-1] for _ in data_train])
	
	return X_test, y_test


def load_in_torch_fmt(X_train, y_train):

	X_train = torch.from_numpy(X_train).long()
	y_train = torch.from_numpy(y_train).long()

	return X_train, y_train



def save_checkpoint(state,is_best,OutputDir,test_index):
	if is_best:
		print('=> Saving a new best from epoch %d"' % state['epoch'])
		torch.save(state, OutputDir + '/' + str(test_index) +'_checkpoint.pth.tar')
		
	else:
		print("=> Validation Performance did not improve")
		
def ytest_ypred_to_file(y, y_pred, out_fn):
	with open(out_fn,'w') as f:
		for i in range(len(y)):
			f.write(str(y[i])+'\t'+str(y_pred[i])+'\n')
			
			
def chunkIt(seq, num): 
	avg = len(seq) / float(num)
	out = []
	last = 0.0
	
	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	return out

def shuffleData(X, y):
	index = [i for i in range(len(X))]
	# np.random.seed(42)
	random.shuffle(index)
	new_X = X[index]
	new_y = y[index]
	return new_X, new_y

def round_pred(pred):
	list_result = []
	for i in pred:
		if i >0.3:
			list_result.append(1)
		elif i <=0.3:
			list_result.append(0)
	return torch.tensor(list_result)
	
def get_loss(logits, label, criterion):
	loss = criterion(fx.squeeze(), train_y.type(torch.FloatTensor).to(device))
	loss = (loss.float()).mean()
	# flooding method
	loss = (loss - 0.06).abs() + 0.06

	# multi-sense loss
	# alpha = -0.1
	# loss_dist = alpha * cal_loss_dist_by_cosine(model)
	# loss += loss_dist

	return loss
	

def get_loss2(fx, train_y, criterion):
	loss = criterion(fx, train_y)
	loss = (loss.float()).mean()
	# flooding method
	loss = (loss - 0.06).abs() + 0.06

	# multi-sense loss
	# alpha = -0.1
	# loss_dist = alpha * cal_loss_dist_by_cosine(model)
	# loss += loss_dist

	return loss

def adjust_model(model):
	# Freeze some layers
	# util_freeze.freeze_by_names(model, ['embedding', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge', 'soft_attention', 'layers'])
	# util_freeze.freeze_by_names(model, ['embedding', 'embedding_merge'])
	# util_freeze.freeze_by_names(model, ['embedding_merge'])
	# util_freeze.freeze_by_names(model, ['embedding'])
	
	# unfreeze some layers
	for name, child in model.named_children():
	    for sub_name, sub_child in child.named_children():
	        if name == 'layers' and (sub_name == '3'):
	            print('Encoder Is Unfreezing')
	            for param in sub_child.parameters():
	                param.requires_grad = True
	
	print('-' * 50, 'Model.named_parameters', '-' * 50)
	for name, value in model.named_parameters():
		print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))
	
	# Count the total parameters
	params = list(model.parameters())
	k = 0
	for i in params:
		l = 1
		for j in i.size():
			l *= j
		k = k + l
	print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)
	pass
	
#--------------------------------------------------->>>
class BahdanauAttention(nn.Module):
	"""
	input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
									h_n: (num_directions, batch_size, units)
	return: (batch_size, num_task, units)
	"""
	def __init__(self,in_features, hidden_units,num_task):
		super(BahdanauAttention,self).__init__()
		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

	def forward(self, hidden_states, values):
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
		#print(attention_weights.shape,values.shape)
		context_vector = torch.matmul(values,attention_weights)
		context_vector = torch.transpose(context_vector,1,2)
		return context_vector, attention_weights
		
class Embedding(nn.Module):
	def __init__(self, vocab_size,d_model,max_len):
		super(Embedding, self).__init__()
		self.tok_embed = nn.Embedding(vocab_size, d_model)
		self.pos_embed = nn.Embedding(max_len, d_model)
		self.norm = nn.LayerNorm(d_model)

	def forward(self, x):
		seq_len = x.size(1)
		pos = torch.arange(seq_len, device=device, dtype=torch.long)
		pos = pos.unsqueeze(0).expand_as(x)

		embedding = self.pos_embed(pos)
		embedding = embedding + self.tok_embed(x)
		embedding = self.norm(embedding)
		return embedding


class Adapt_emb_CNNLSTM_ATT(nn.Module):
	def __init__(self):
		super(Adapt_emb_CNNLSTM_ATT, self).__init__()
		kernel_size = 10
		max_len = 33
		d_model = 32
		vocab_size = 28
	
		self.embedding = Embedding(vocab_size, d_model, max_len)
		
		self.conv1 = nn.Sequential( 
			nn.Conv1d(
					in_channels=32,      # input height
					out_channels=256,    # n_filters
					kernel_size = kernel_size),
					# dilation =2),    
					# padding = int(kernel_size/2)),
			# padding=(kernel_size-1)/2 
			nn.ReLU(),    # activation
			nn.MaxPool1d(kernel_size=2),
			nn.BatchNorm1d(256),
			nn.Dropout())
		
		self.lstm = torch.nn.LSTM(256, 128, 1, batch_first=True, bidirectional=True)
		self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=1)
	
		self.fc_task = nn.Sequential(
			nn.Linear(256, 32),
			nn.Dropout(0.7),
			nn.ReLU(),
			nn.Linear(32, 2),
		)
		self.classifier = nn.Linear(2, 1)
	
	#---------------->>>
	def forward(self, x):
		x = self.embedding(x) 
		x = x.transpose(1, 2)
		batch_size, features, seq_len = x.size()
	
		x = self.conv1(x) 
		x = x.transpose(1, 2)
		#rnn layer
		out, (h_n, c_n) = self.lstm(x) 
		h_n = h_n.view(batch_size, out.size()[-1])
		context_vector, attention_weights = self.Attention(h_n, out)
		reduction_feature = self.fc_task(torch.mean(context_vector,1))

		representation = reduction_feature 
		logits_clsf = self.classifier(representation)
		logits_clsf1 = logits_clsf
		logits_clsf = torch.sigmoid(logits_clsf)  

		return logits_clsf, representation
		

#--------------------------------------------------->>>
def calculateScore(y, pred_y):
	
	y = y.data.cpu().numpy()
	tempLabel = np.zeros(shape = y.shape, dtype=np.int32)
	
	for i in range(len(y)):
		if pred_y[i] < 0.3:
			tempLabel[i] = 0;
		else:
			tempLabel[i] = 1;
			
	accuracy = metrics.accuracy_score(y, tempLabel)
	
	confusion = confusion_matrix(y, tempLabel)
	TN, FP, FN, TP = confusion.ravel()
	
	sensitivity = recall_score(y, tempLabel)
	specificity = TN / float(TN+FP)
	MCC = matthews_corrcoef(y, tempLabel)
	
	F1Score = (2 * TP) / float(2 * TP + FP + FN)
#	precision = TP / float(TP + FP)
	
	precision = metrics.precision_score(y, tempLabel, pos_label=1) 
	recall = metrics.recall_score(y, tempLabel, pos_label=1)
	
	pred_y = pred_y.reshape((-1, ))
	
	# ROCArea = roc_auc_score(y, pred_y)
	ROCArea = metrics.roc_auc_score(y, pred_y)
	fpr, tpr, thresholds = roc_curve(y, pred_y)
	
	pre, rec, threshlds = precision_recall_curve(y, pred_y)
	pre = np.fliplr([pre])[0] 
	rec = np.fliplr([rec])[0]  
	AUC_prec_rec = np.trapz(rec,pre)
	AUC_prec_rec = abs(AUC_prec_rec)
	
	print('sn' , sensitivity, 'sp' , specificity, 'acc' , accuracy, 'MCC' , MCC, 'AUC' , ROCArea,'precision' , precision, 'F1' , F1Score)
	
	return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea,'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds,'pre_recall_curve':AUC_prec_rec,'prec':pre,'reca':rec}


# from scipy import interp
import matplotlib.pyplot as plt


def analyze(temp, OutputDir,species):
	
	trainning_result, validation_result, testing_result = temp;
	
	#写文件
	file = open(OutputDir + '/{}_performance.txt'.format(species), 'w')
	
	index = 0
	for x in [trainning_result, validation_result, testing_result]:
		
		title = ''
		if index == 0:
			title = 'training_{}_'.format(species)
		if index == 1:
			title = 'validation_{}_'.format(species)
		if index == 2:
			title = 'testing_{}_'.format(species)
			
		index += 1;
		
		file.write(title +  'results\n') 
		
		for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1']:
			
			total = [] 
			
			for val in x:
				total.append(val[j])
			file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n') 
			
		file.write('\n\n______________________________\n') 
	file.close();
	
	# plot
	index = 0
	for x in [trainning_result, validation_result, testing_result]:
		
		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)
		
		#************************** ROC Curve*********************************
		i = 0
		for val in x: # 10个{}
			tpr = val['tpr']
			fpr = val['fpr']
			tprs.append(np.interp(mean_fpr, fpr, tpr))
			tprs[-1][0] = 0.0
			roc_auc = auc(fpr, tpr)
			aucs.append(roc_auc)
			plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc)) 
		
			i += 1
		
		print;
		
		#
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8) 
		
		#
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		plt.plot(mean_fpr, mean_tpr, color='b',
				label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
				lw=2, alpha=.8)
		
		#
		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.') 
		#
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic curve')
		plt.legend(loc="lower right")
		
		title = ''
		if index == 0:
			title = 'training_'
		if index == 1:
			title = 'validation_'
		if index == 2:
			title = 'testing_'
			
		plt.savefig( OutputDir + '/' + title +'ROC.png')
		plt.close('all');
	
		#************************** Precision Recall Curve*********************************
		i = 0
		prs = []
		pre_aucs = []
		mean_recal= np.linspace(0, 1, 100)
		for val in x:
			pre = val['prec']
			rec = val['reca']
			prs.append(np.interp(mean_recal, rec, pre))
			prs[-1][0] = 0.0
			p_r_auc = auc(rec, pre)
			pre_aucs.append(p_r_auc)
			plt.plot(rec, pre, lw=1, alpha=0.3,label='PRC fold %d (AUC = %0.2f)' % (i+1, p_r_auc))
			
			i += 1
			
		print;
		
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)
		
		mean_pre = np.mean(prs, axis=0)
		mean_pre[-1] = 1.0
		mean_auc = auc(mean_recal, mean_pre)
		std_auc = np.std(pre_aucs)
		plt.plot(mean_recal, mean_pre, color='b',
				label=r'Mean PRC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
				lw=2, alpha=.8)
		
		std_pre = np.std(prs, axis=0)
		pre_upper = np.minimum(mean_pre + std_pre, 1)
		pre_lower = np.maximum(mean_pre - std_pre, 0)
		plt.fill_between(mean_recal, pre_lower, pre_upper, color='grey', alpha=.2,
						label=r'$\pm$ 1 std. dev.')
		
		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision Recall curve')
		plt.legend(loc="lower right")
		
		title = ''
		
		if index == 0:
			title = 'training_'
		if index == 1:
			title = 'validation_'
		if index == 2:
			title = 'testing_'
			
		plt.savefig( OutputDir + '/' + title +'Pre_R_C.png')
		plt.close('all')
		index += 1;
	
def get_k_fold_data(k, i, X, y): 
	
	fold_size = X.shape[0] // k 
	
	val_start = i * fold_size
	if i != k - 1:
		val_end = (i + 1) * fold_size
		X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
		X_train = np.concatenate([X[0:val_start], X[val_end:]], 0)
		y_train = np.concatenate([y[0:val_start], y[val_end:]], 0)
	else:
		X_valid, y_valid = X[val_start:], y[val_start:]   
		X_train = X[0:val_start]
		y_train = y[0:val_start]
	
	return X_train, y_train, X_valid, y_valid


if __name__ == '__main__':
	# Hyper Parameters------------------>>
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)
	np.random.seed(42)
	random.seed(42)
	torch.backends.cudnn.deterministic = True
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	device_cpu = torch.device("cpu")
	# wordvec_len =4
	
	EPOCH = 50
	BATCH_SIZE = 256
	LR = 0.001
	kernel_size = 32
	
	#Model
	net = Adapt_emb_CNNLSTM_ATT().to(device)

	print(net)  # print the net architecture of model loaded in the previous step
	adjust_model(net)

	trainning_result = []
	validation_result = []
	testing_result = []
	
	train_loss_sum, valid_loss_sum, test_loss_sum= 0, 0, 0
	train_acc_sum , valid_acc_sum , test_acc_sum = 0, 0, 0
	
	
	test_acc = []    
	test_auc = []
	test_losses = []  
	
	#-------------------------------------------------->>>
	#fold: 
	k_folds = 5 # you may change the number of fold according to your case
	
	if k_folds == -1:       					
		all_index = ['-1']

	elif k_folds != -1 and k_folds != 0: 
		all_index = [i for i in range(k_folds)]

	for index_fold in all_index: #index_fold #test_index
		X_train, y_train, X_valid, y_valid = [],[],[],[]

		#------------load training dataset files---------------------->>
		train_pos_fa = './data/mm10/mm10_positive_training.fa'
		train_neg_fa = './data/mm10/mm10_negative_training.fa'

		#------------set output path---------------------->>
		OutputDir = './Result/Adapt_Train_mm10/fold_{0}'.format(index_fold)
		
		OutputDir_tsne_pca = OutputDir +'/Out_tsne_pca'
		if os.path.exists(OutputDir_tsne_pca):
			print('OutputDir is exitsted')
		else:
			os.makedirs(OutputDir_tsne_pca)
			print('success create dir test')
			
			
		
		if k_folds == -1: 
			X_train, y_train, X_valid, y_valid = load_train_val_bicoding(train_pos_fa,train_neg_fa)
		
		elif k_folds != -1 and k_folds != 0: 
			train_all_X, train_all_y = load_test_bicoding(train_pos_fa, train_neg_fa)
			X_train, y_train, X_valid, y_valid = get_k_fold_data(k_folds, int(index_fold), train_all_X, train_all_y)

		rus = RandomUnderSampler(random_state=42)
    	X_train, y_train = rus.fit_resample(X_train, y_train)

    	# Balance data
    	rus_indep = RandomUnderSampler(random_state=42,sampling_strategy=0.2)
    	X_valid, y_valid = rus_indep.fit_resample(X_valid, y_valid)
		
		X_train, y_train = load_in_torch_fmt(X_train, y_train)
		X_valid, y_valid = load_in_torch_fmt(X_valid, y_valid)
		
		
		#------------------->>>
		print('Train data: ',X_train.shape[0])
		print('Valid data: ',X_valid.shape[0])
		# ---------------------------------->>
		
		train_loader = Data.DataLoader(Data.TensorDataset(X_train,y_train), BATCH_SIZE, shuffle = False)
		val_loader = Data.DataLoader(Data.TensorDataset(X_valid, y_valid), BATCH_SIZE, shuffle = False)


		model = net

		criterion = nn.BCELoss(size_average=False)												
		# criterion = nn.CrossEntropyLoss()														
		# criterion = nn.BCEWithLogitsLoss(size_average=False)	
		# optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
		optimizer = torch.optim.AdamW(params = model.parameters(), lr=LR, weight_decay=0.0025)
		
		train_losses = []
		val_losses = []   
		
		train_acc = []   
		val_acc = []      

		best_acc = 0
		patience = 0
		patience_limit = 20
		
		epoch_list = [] 
		torch_val_best, torch_val_y_best  = torch.tensor([]),torch.tensor([]) 
		
		for epoch in range(EPOCH):
			repres_list, label_list = [],[]
			
			torch_train, torch_train_y = torch.tensor([]),torch.tensor([])
			torch_val, torch_val_y     = torch.tensor([]),torch.tensor([])
			torch_test, torch_test_y   = torch.tensor([]),torch.tensor([])
			
			model.train() #Train
			# scheduler.step()
			correct = 0 
			train_loss = 0
			for step, (train_x, train_y) in enumerate(train_loader):

				train_x = Variable(train_x, requires_grad=False).to(device) 
				train_y = Variable(train_y, requires_grad=False).to(device) 
				fx, presention = model(train_x)  #torch.Size([256, 1])
				loss = criterion(fx.squeeze(), train_y.type(torch.FloatTensor).to(device))   	

				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				repres_list.extend(presention.cpu().detach().numpy())
				label_list.extend(train_y.cpu().detach().numpy())
				
				
				pred = round_pred(fx.data.cpu().numpy()).to(device)						
				correct += pred.eq(train_y.view_as(pred)).sum().item()
				
				train_loss += loss.item() * len(train_y)
				
				# pred_prob = fx[:,1] 	        #CrossEntropyLoss case								
				pred_prob = fx					#BCELoss case				    
				torch_train = torch.cat([torch_train,pred_prob.data.cpu()],dim=0)
				torch_train_y = torch.cat([torch_train_y,train_y.data.cpu()],dim=0)
				
				if (step+1) % 10 == 0:
					print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'%(epoch + 1, EPOCH, 
																		step + 1, len(X_train)//BATCH_SIZE, 
																		loss.item()))
			
			train_losses.append(train_loss/len(X_train))
			epoch_list.append(epoch) 
																	
			accuracy_train = 100.*correct/len(X_train)
			#：Epoch: 1, Loss: 0.64163, Training set accuracy: 908/1426 (63.675%)
			print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
				epoch + 1, loss.item(), correct, len(X_train), accuracy_train))
			train_acc.append(accuracy_train)
			
			model.eval() #Valid
			val_loss = 0
			correct = 0
			repres_list_valid , label_list_valid = [],[]
			
			with torch.no_grad():
				for step, (valid_x, valid_y) in enumerate(val_loader):     #val_loader
					valid_x = Variable(valid_x, requires_grad=False).to(device)  
					valid_y = Variable(valid_y, requires_grad=False).to(device) 
					
					optimizer.zero_grad()  #--->>
					y_hat_val, presention_valid = model(valid_x)
					# loss = criterion(y_hat_val, valid_y.to(device)).item()      # batch average loss		  #CrossEntropyLoss case
					loss = criterion(y_hat_val.squeeze(), valid_y.type(torch.FloatTensor).to(device)).item()  #BCELoss case
					val_loss += loss * len(valid_y)             # sum up batch loss 
					
					#加入列表用于PCA/tsne  #valid
					repres_list_valid.extend(presention_valid.cpu().detach().numpy())
					label_list_valid.extend(valid_y.cpu().detach().numpy())
					

					pred_val = round_pred(y_hat_val.data.cpu().numpy()).to(device)          #BCELoss case
					
					# get the index of the max log-probability
					correct += pred_val.eq(valid_y.view_as(pred_val)).sum().item()
					
					# pred_prob = y_hat.max(1, keepdim = True)[0]
					# pred_prob_val = y_hat_val[:,1] 					#CrossEntropyLoss case
					pred_prob_val = y_hat_val											#BCELoss case
					torch_val = torch.cat([torch_val,pred_prob_val.data.cpu()],dim=0)
					torch_val_y = torch.cat([torch_val_y,valid_y.data.cpu()],dim=0)

			val_losses.append(val_loss/len(X_valid)) # all loss / all sample
			accuracy_valid = 100.*correct/len(X_valid)

			print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
				val_loss/len(X_valid), correct, len(X_valid), accuracy_valid))
				
			val_acc.append(accuracy_valid)
			
			#----------- saving record -------------------------------------->>>
			cur_acc = accuracy_valid
			is_best = bool(cur_acc >= best_acc) 
			best_acc = max(cur_acc, best_acc)  
			
			if is_best: 
				torch_val_best = torch_val 
				torch_val_y_best = torch_val_y
			

			save_checkpoint({
				'epoch': epoch+1,
				'state_dict': model.state_dict(),
				'best_accuracy': best_acc,
				'optimizer': optimizer.state_dict()
			}, is_best,OutputDir,index_fold)
			
			#patience 
			if not is_best: 
				patience+=1
				if patience >= patience_limit:
					break
			else:
				patience = 0
			print('> best acc:',best_acc)


	
	
	
	
	
	
