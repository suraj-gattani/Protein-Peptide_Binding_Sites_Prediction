import numpy as np
import pandas as pd
import pickle as pk
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef


def main():
		# read the training data file for window size 1
	train_df_1 = pd.read_csv('feature_file_train_ws1.csv', header=None)
	train_1 = train_df_1.as_matrix()
	y_1 = train_1[:,0]
	X_1 = train_1[:,1:]
	scaler = StandardScaler()
	X_scale_1 = scaler.fit_transform(X_1)

	# read the test data file for window size 1
	test_df_1 = pd.read_csv('feature_file_test_ws1.csv', header=None)
	test_1 = test_df_1.as_matrix()
	y = test_1[:,0]
	X_test_1 = test_1[:,1:]
	X_scale_1_test = scaler.transform(X_test_1)

	param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective': 'multi:softprob','num_class': 2,'n_estimators':100,'min_child_weight':5,'subsample':0.9}
	#res = xgb.train(param, dtrain, num_boost_round=10, nfold=5,metrics={'error'}, seed=0)
	
	clf = xgb.XGBClassifier(**param)
	clf.fit(X_scale_1,y_1)
	predicted_train=clf.predict(X_scale_1)
	predicted=clf.predict(X_scale_1_test)
	proba_train=clf.predict_proba(X_scale_1)
	proba_test=clf.predict_proba(X_scale_1_test)
	

	np.set_printoptions(threshold=np.nan)
	print('Predicted probabilities for train = ')
	print(np.matrix(proba_train))
	print("")
	print('Predicted probabilities for test = ' )
	print(np.matrix(proba_test))
	print("")
	print('BAG_train')
	#clf = svm.SVC()
	#predicted = cross_val_predict(clf, X, y, cv=10)
	confusion = confusion_matrix(y_1, predicted_train)
	print(confusion)
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
	MCC_cla = matthews_corrcoef(y_1, predicted_train)
	F1_cla = f1_score(y_1, predicted_train)
	PREC_cla = precision_score(y_1, predicted_train)
	REC_cla = recall_score(y_1, predicted_train)
	Accuracy_cla = accuracy_score(y_1, predicted_train)
	print('TP = ', TP)
	print('TN = ', TN)
	print('FP = ', FP)
	print('FN = ', FN)
	print('Recall/Sensitivity = %.5f' %REC_cla)
	print('Specificity = %.5f' %SPE_cla)
	print('Accuracy_Balanced = %.5f' %ACC_Bal)
	print('Overall_Accuracy = %.5f' %Accuracy_cla)
	print('FPR_bag = %.5f' %FPR)
	print('FNR_bag = %.5f' %FNR)
	print('Precision = %.5f' %PREC_cla)
	print('F1 = %.5f' % F1_cla)
	print('MCC = %.5f' % MCC_cla)
	print("")
	print("BAG_test")

	#clf = svm.SVC()
	#predicted = cross_val_predict(clf, X, y, cv=10)

	confusion = confusion_matrix(y, predicted)
	print(confusion)
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
	print('TP = ', TP)
	print('TN = ', TN)
	print('FP = ', FP)
	print('FN = ', FN)
	print('Recall/Sensitivity = %.5f' %REC_cla)
	print('Specificity = %.5f' %SPE_cla)
	print('Accuracy_Balanced = %.5f' %ACC_Bal)
	print('Overall_Accuracy = %.5f' %Accuracy_cla)
	print('FPR_bag = %.5f' %FPR)
	print('FNR_bag = %.5f' %FNR)
	print('Precision = %.5f' %PREC_cla)
	print('F1 = %.5f' % F1_cla)
	print('MCC = %.5f' % MCC_cla)


if __name__ == '__main__':
    main()