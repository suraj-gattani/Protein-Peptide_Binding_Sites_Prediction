# coding: utf-8

import numpy as np
import pandas as pd
import pickle as pk
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef

def main():
	# read the training data file
	train_df = pd.read_csv('feature_file.csv', header=None, low_memory=False)

	# column 0 is class, 1 represents disulfide bonding exist whereas -1 represents disulfide does not exist
	train = train_df.as_matrix()

	y = train[:,0]
	X = train[:,1:]

	param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective': 'multi:softprob','num_class': 2,'n_estimators':100,'min_child_weight':5,'subsample':0.9}
	#res = xgb.train(param, dtrain, num_boost_round=10, nfold=5,metrics={'error'}, seed=0)
	
	clf = xgb.XGBClassifier(**param)
	predicted = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)
	
	confusion = confusion_matrix(y, predicted)
	#print(confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	# Specificity
	SPE_cla = (TN/float(TN+FP))

	# False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	# compute MCC
	MCC_cla = matthews_corrcoef(y, predicted)
	F1_cla = f1_score(y, predicted)
	PREC_cla = precision_score(y, predicted)
	REC_cla = recall_score(y, predicted)
	Accuracy_cla = accuracy_score(y, predicted)
	fpr, tpr, _ = roc_curve(y, predicted)
	print('Results:, %.5f' %auc(fpr,tpr),',%.5f' %ACC_Bal,',%.5f' % MCC_cla)


if __name__ == '__main__':
    main()







