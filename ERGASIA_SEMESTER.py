#Εισαγωγή των κατάλληλων βιβλιοθηκών
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from IPython.display import Image
from os import system
import os
from statistics import mean 
from mlxtend.plotting import plot_confusion_matrix

from sklearn import decomposition, tree
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import LeavePGroupsOut, train_test_split, RepeatedStratifiedKFold, LeaveOneGroupOut, GridSearchCV, ShuffleSplit, GroupShuffleSplit
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings

#Εντολή για να μην εμφανίζονται οι πρειδοποιήσεις
warnings.filterwarnings("ignore")

#Διαβάζω το dataset
df = pd.read_csv('./parkinsons.data')

#Bλέπω από τη βιβλιογραφία την περιγραφή του dataset
description = open('./parkinsons.names', 'r+')
features = {'name':'ASCII subject name and recording number',
			'MDVP:Fo(Hz)':'Average vocal fundamental frequency',
			'MDVP:Fhi(Hz)':'Maximum vocal fundamental frequency',
			'MDVP:Flo(Hz)':'Minimum vocal fundamental frequency',
			'MDVP:Jitter(%)':'Measure of variation in fundamental frequency',
			'MDVP:Jitter(Abs)':'Measure of variation in fundamental frequency',
			'MDVP:RAP':'Measure of variation in fundamental frequency',
			'MDVP:PPQ':'Measure of variation in fundamental frequency',
			'Jitter:DDP':'Measure of variation in fundamental frequency',
			'MDVP:Shimmer':'Measure of variation in amplitude',
			'MDVP:Shimmer(dB)':'Measure of variation in amplitude',
			'Shimmer:APQ3':'Measure of variation in amplitude',
			'Shimmer:APQ5':'Measure of variation in amplitude',
			'MDVP:APQ':'Measure of variation in amplitude',
			'Shimmer:DDA':'Measure of variation in amplitude',
			'NHR':'Two measures of ratio of noise to tonal components in the voice',
			'HNR':'Two measures of ratio of noise to tonal components in the voice',
			'status':'Health status of the subject (one) for Parkinson, and (zero) for healthy',
			'RPDE':'Nonlinear dynamical complexity measure',
			'D2':'Nonlinear dynamical complexity measure',
			'DFA':'Signal fractal scaling exponent',
			'spread1':'Nonlinear measure of fundamental frequency variation',
			'spread2':'Nonlinear measure of fundamental frequency variation',
			'PPE':'Nonlinear measure of fundamental frequency variation'}

features = pd.DataFrame(data=features, index=[0])

#Συνάρτηση όπου δείχνει την περιγραφή του κάθε χαρακτηριστικού
def feat_info(col_name):
	print(features[col_name].loc[0])

feat_info('PPE')

targets_class = df.loc[:, 'status'].values

#Δημιουργώ το φάκελο Plots, για την αποθήκευση των γραφήματων
plots = os.path.join(os.getcwd(), 'Plots')
if not os.path.isdir(plots):
	os.mkdir('Plots')

#Συνάρτηση για την ανάλυση του dataset
def check_the_dataframe(df):
	print(df.shape)
	print(df.groupby('status').count())
	print(df.head(10))
	print(df.isnull().sum())
	print(df.info())
	print(df.iloc[:,:][~df.iloc[:,1:].applymap(np.isreal).all(1)])
	print(df.describe().transpose())
	plt.figure(figsize=(15,15))
	sns.heatmap(df.corr(), annot=True)
	plt.savefig(os.path.join(plots, 'Heatmap.png'))
	plt.show()

check_the_dataframe(df)

#Συνάρτηση για την ανάλυση των κλάσεων-στόχων του dataset
def check_target_classes():
	labels = ['Healthy', 'Parkinson']
	fig = plt.figure()
	ax1 = sns.countplot(x = 'status', data = df)
	ax1.set_title('Number of Instances per category')
	ax1.set_xticklabels(['Healthy', 'Parkinson'])
	ax1.set_xlabel('Status')
	ax1.set_ylabel('Counts')
	for p in ax1.patches:
		ax1.annotate('{:.0f} voice instances'.format(p.get_height()), (p.get_x()+0.12, p.get_height()+2))


	fig.savefig(os.path.join(plots, 'Target_Classes.png'))
	plt.show()

	print("Ο αριθμός των φωνητικών μετρήσεων από ανθρώπους με Parkinson είναι:", np.sum(targets_class == 1), 
	", σε ποσοστό",round(np.sum(targets_class == 1)/(np.sum(targets_class == 0) + np.sum(targets_class == 1))*100, 2),"%")
	print("Ο αριθμός των υγιών φωνητικών μετρήσεων είναι: ", np.sum(targets_class == 0), 
	", σε ποσοστό",round(np.sum(targets_class == 0)/(np.sum(targets_class == 0) + np.sum(targets_class == 1))*100, 2),"%")


	plt.close()

check_target_classes()


#Συνάρτηση για PCA ανάλυση και αποθηκέυω τα γραφήματα στο φάκελο που δημιουργώ με όνομα Plots
def PCA_(df):
	#Προετοιμάζω τα δεδομενα για την PCA ανάλυση
	pca = decomposition.PCA()
	pca.n_components = 22
	sc_data = StandardScaler().fit_transform(df.drop(['name', 'status'], axis = 1))

	pca_data = pca.fit_transform(sc_data)

	#Πλοτάρω τη συνολική πολυπλοκότητα του dataset, ως προς το πλήθος των χαρακτηριστικών.
	percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
	cum_var_explained = np.cumsum(percentage_var_explained)
	plt.figure(1, figsize=(6, 4))
	plt.clf()
	plt.plot(cum_var_explained, linewidth=2)
	plt.axis('tight')
	plt.grid()
	plt.xlabel('n_components')
	plt.ylabel('Cumulative Variance (%)')
	plt.savefig(os.path.join(plots, 'Cumulative_Variance.png'))
	plt.close()

	#Τα πλοτάρω σε 2 διαστάσεις
	pca.n_components = 2
	pca_data = pca.fit_transform(sc_data)
	pca_data = np.vstack((pca_data.T, df['status'])).T
	pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "status"))
	sns.FacetGrid(pca_df, hue="status", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend(labels=['Healthy', 'Parkinson'])
	plt.savefig(os.path.join(plots, '2D.png'))
	plt.close()

	#Έπειτα και στις 3 διαστάσεις
	pca.n_components = 3
	pca_data = pca.fit_transform(sc_data)
	pca_data = np.vstack((pca_data.T, df['status'])).T
	pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "3rd_principal", "status"))
	pca_df_healthy = pca_df[pca_df['status']==0]
	pca_df_parkinson = pca_df[pca_df['status']==1]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pca_df_healthy['1st_principal'], pca_df_healthy['2nd_principal'], pca_df_healthy['3rd_principal'], c='blue', s=20, marker = 'o', label='Healthy')
	ax.scatter(pca_df_parkinson['1st_principal'], pca_df_parkinson['2nd_principal'], pca_df_parkinson['3rd_principal'], c='orange', s=20, marker = 'o', label='Parkinson')
	
	ax.view_init(30, 185)
	ax.legend()
	plt.savefig(os.path.join(plots, '3D.png'))
	plt.close()


PCA_(df) #Συνάρτηση όπου υλοποιώ PCA ανάλυση.

#Συνάρτηση για την παρουσίαση της συνάρτησης με συντελεστές όπως προκύπτουν απο τη Lasso ομαλοποίηση
def print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))

    return " + ".join("%s * %s" % (round(coef, 4), name) for coef, name in lst)

#Συνάρτηση όπου κάνω επιλογή χαρακτηριστικών με Lasso ομαλοποίηση.
def feature_selection(df):
	X = df.drop(['name', 'status'], axis = 1)
	# print(X.shape)
	y = df['status']

	scaler = StandardScaler()
	scaler.fit(X)
	X_scaled = scaler.transform(X)

	selector = SelectFromModel(LassoCV())
	selector.fit(X_scaled, y)
	print(selector.get_support())

	coef = pd.Series(selector.estimator_.coef_, index = X.columns)
	imp_coef = coef.sort_values()
	print(coef, imp_coef != 0)
	plt.figure(figsize = (8.0, 15.0))
	imp_coef.plot(kind = "barh")
	plt.title("Feature importance using Lasso regularization")
	plt.savefig(os.path.join(plots, 'Feature_selection.png'))
	plt.show()
	print(selector.transform(X_scaled).shape)

	# print(coef.index[imp_coef != 0])
	print ("Lasso model:", print_coefs(selector.estimator_.coef_))
	# new_df = pd.DataFrame(df[coef.index[imp_coef != 0]], columns=coef.index[imp_coef != 0]) #Εδώ δημιουργώ και το καινούργιο DataFrame με τα καινούργια 
																							  #χαρακτηριστικά.

	return selector.transform(X_scaled), y


X, y = feature_selection(df)


print(X.shape)



#####################################################################################################
##############################################   ML ALGORTITHMS   ###################################
#####################################################################################################

#Σε κάθε αλγόριθμο, τον έχω υλοποιήσει ως συνάρτηση, όπου σε κάθε συνάρτηση 
#δημιουργώ ένα φάκελο για να αποθηκεύω τα αρχεία όπου εξάγονται από κάθε συνάρτηση.

#Τα αρχεία που υλοποιούνται από κάθε συνάρτηση είναι τα confusion matrix από κάθε
#αλγόριθμο στο train_set και test_set. Επιπλέον, σε κάθε συνάρτηση κάνω GridSearchCV,
#με τη παράμετρο cv ίση με lpgo, όπου είναι η μεταβλητή για τη μέθοδο LeaveOneGroupOut(),
#για να υλοποιώ GridSearchCV, αφήνοοντας ένα άτομο έξω. 

#Σε κάθε αλγόριθμο υλοποιείται
#GridSearchCV, με τις αντιστοιχες παραμέτρους, με τη σημείωση ότι η παράμετρος class_weight
#σε όλες τις συναρτήσεις να εξετάζεται στο ίδιο εύρος τιμών.

#Στο DecisionTree ταξινομητή υλοποιώ και το δέντρο που εξάγεται.

DT_results = os.path.join(os.getcwd(), 'DT_results')
if not os.path.isdir(DT_results):
	os.mkdir('DT_results')

def DecisionTree(X_train, y_train, X_test, y_test, groups_train, lpgo):
	tree_params = {'criterion':['gini','entropy'],
				   'max_depth':[2, 5, 10, 30, 50, 70, 100],
				   'max_features': [1, 5, 10, 20, 50, 'None'],
				   'class_weight':[{0:10, 1:1}, {0:1, 1:1}, {0:1, 1:3}, {0:1, 1:10}, 'balanced'],
				   'ccp_alpha':[0.0, 0.2, 0.4, 0.6, 0.8, 1]}
	
	clf = GridSearchCV(DecisionTreeClassifier(), tree_params, scoring = 'accuracy', cv=lpgo, n_jobs=-1)
	clf.fit(X_train, y_train, groups_train)

	
	print('######################################################################################################################')
	print('######                                DECISION TREE MODEL ({}oς διαχωρισμος)                                      ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the Decision Tree model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precision in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(DT_results, str(i)+'_Training_DT.png'))

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precison in testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(DT_results, str(i)+'_Testing_DT.png'))

	fig = plt.figure(figsize=(10,10))
	_ = tree.plot_tree(clf.best_estimator_, 
                   #feature_names=iris.feature_names,  
                   #class_names=iris.target_names,
                   filled=True)
	plt.savefig(os.path.join(DT_results, str(i)+'PARKINSON_TREE.png'))

    
	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)


	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_



GNB_results = os.path.join(os.getcwd(), 'GNB_results')
if not os.path.isdir(GNB_results):
	os.mkdir('GNB_results')

def Gaussian(X_train, y_train, X_test, y_test, groups_train, lpgo):
	params = {'var_smoothing': np.logspace(0,-9, num=100)}

	clf = GridSearchCV(GaussianNB(), param_grid=params, cv=lpgo, scoring='accuracy', n_jobs=-1)

	clf.fit(X_train, y_train, groups_train)

	print('######################################################################################################################')
	print('######                                GAUSSIAN_NB MODEL ({}oς διαχωρισμος)                                       ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the Naive Bayes model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precision in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(GNB_results, str(i)+'_Training_GNB.png'))
	plt.close()

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precision in testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(GNB_results, str(i)+'_Testing_GNB.png'))
	plt.close()
    
	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)


	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_



LogReg_results = os.path.join(os.getcwd(), 'LogReg_results')
if not os.path.isdir(LogReg_results):
	os.mkdir('LogReg_results')

def LogReg(X_train, y_train, X_test, y_test, groups_train, lpgo):
	tuned_parameters = [{'C': [10**-4, 10**-2, 10**0, 10**2, 10**4],
						'tol':[10**-8, 10**-6, 10**-4, 10**-2],
						'max_iter':[100, 1000],
						'penalty':['l1', 'l2', None], 
						'class_weight':[{0:10, 1:1}, {0:1, 1:1}, {0:1, 1:3}, {0:1, 1:10}, 'balanced'],
						'solver':['lbfgs', 'liblinear', 'saga']
						}]

	clf = GridSearchCV(LogisticRegression(), param_grid=tuned_parameters, cv=lpgo, scoring='accuracy', n_jobs=-1)

	clf.fit(X_train, y_train, groups_train)

	print('######################################################################################################################')
	print('######                                LOGISTIC REGRESSION MODEL ({}oς διαχωρισμος)                                ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the Logistic model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precion in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(LogReg_results, str(i)+'_Training_LogReg.png'))
	plt.close()

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precion testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(LogReg_results, str(i)+'_Testing_LogReg.png'))
	plt.close()

	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)


	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_


RFC_results = os.path.join(os.getcwd(), 'RFC_results')
if not os.path.isdir(RFC_results):
	os.mkdir('RFC_results')

def RFC(X_train, y_train, X_test, y_test, groups_train, lpgo):
	param_grid = {'bootstrap': [True, False],
					'oob_score': [True, False],
					'max_depth': [2, 5, 10],
					'max_features': ['auto', 'sqrt'], 
# 					'min_samples_leaf': [1, 2, 4], 
					'min_samples_split': [2, 5, 10], 
					'n_estimators': [100, 300, 500],
					'criterion': ['gini', 'entropy'],
					'class_weight':[{0:10, 1:1}, {0:1, 1:1}, {0:1, 1:3}, {0:1, 1:10}, 'balanced']}

          
	clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=lpgo, scoring='accuracy', n_jobs=-1)

	clf.fit(X_train, y_train, groups_train)

	print('######################################################################################################################')
	print('######                                     RANDOM FOREST MODEL ({}oς διαχωρισμος)                                ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the RFC model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precision in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(RFC_results, str(i)+'_Training_RFC.png'))
	plt.close()

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precision in testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(RFC_results, str(i)+'_Testing_RFC.png'))
	plt.close()

	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)


	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_




SVM_results = os.path.join(os.getcwd(), 'SVM_results')
if not os.path.isdir(SVM_results):
	os.mkdir('SVM_results')


def SVM(X_train, y_train, X_test, y_test, groups_train, lpgo):
	C_range = np.logspace(-2, 10, 15)
	gamma_range = np.logspace(-9, 3, 10)
	class_weight = [{0:10, 1:1}, {0:1, 1:1}, {0:1, 1:3}, {0:1, 1:10}, 'balanced']
	kernel = ['rbf']
	param_grid = dict(gamma=gamma_range, C=C_range, class_weight=class_weight, kernel=kernel)
	

	clf = GridSearchCV(SVC(), param_grid, scoring ='accuracy', cv=lpgo, n_jobs=-1)

	clf.fit(X_train, y_train, groups_train)

	print('######################################################################################################################')
	print('######                            SUPPORT VECTOR MACHINE MODEL ({}oς διαχωρισμος)                                ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the SVM model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precision in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(SVM_results, str(i)+'_Training_SVM.png'))
	plt.close()

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precision in testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(SVM_results, str(i)+'_Testing_SVM.png'))
	plt.close()

	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)


	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_




XGB_results = os.path.join(os.getcwd(), 'XGB_results')
if not os.path.isdir(XGB_results):
	os.mkdir('XGB_results')

def XGBC(X_train, y_train, X_test, y_test, groups_train, lpgo):
	param_grid_gb = {'learning_rate': [0.01,0.1,0.5,0.9],
			  		 'n_estimators' : [5, 50, 100, 150, 200],
			  		 'subsample' : [0.3,0.5,0.9],
			  		 'class_weight' : [{0:10, 1:1}, {0:1, 1:1}, {0:1, 1:3}, {0:1, 1:10}, 'balanced']}

	clf = GridSearchCV(XGBClassifier(), param_grid_gb, scoring = 'accuracy', cv=lpgo, n_jobs=-1)
	clf.fit(X_train, y_train, groups_train)

	print('######################################################################################################################')
	print('######                                       XGB MODEL ({}oς διαχωρισμος)                                        ######'.format(i))
	print('######                                                                                                          ######')
	print('After grid search the SVM model has the follow parameters:')
	print(clf.best_estimator_, clf.best_params_)
	y_predtr = clf.predict(X_train)
	y_predict = clf.predict(X_test)
	print('The accuracy in training set is: {:.2f}%'.format(clf.score(X_train,y_train)*100))
	print('The precision in training set is: {:.2f}%'.format(precision_score(y_train, y_predtr)*100))
	print('The recall in training set is: {:.2f}%'.format(recall_score(y_train, y_predtr)*100))
	print('The F1 score in training set is: {:.2f}%'.format(f1_score(y_train, y_predtr)*100))
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predtr, y_train))
	print(classification_report(y_train, y_predtr, target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predtr, y_train), colorbar=True, show_absolute=True, cmap='Blues')
	labels = ['Healthy', 'Parkinson']
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(XGB_results, str(i)+'_Training_XGB.png'))
	plt.close()

	print("The accuracy in testing set is: {:.2f}%".format(clf.score(X_test,y_test)*100))
	print('The precision in testing set is: {:.2f}%'.format(precision_score(y_test, y_predict)*100))
	print('The recall in testing set is: {:.2f}%'.format(recall_score(y_test, y_predict)*100))
	print("The F1 score in testing set is: {:.2f}%".format(f1_score(y_test, y_predict)*100))
	print('The parameters after the GridSearrch are: ', clf.best_params_)
	print('The confusion matrix and classification report in training set are:')
	print(confusion_matrix(y_predict, y_test))
	print(classification_report(y_test, y_predict))#,target_names = ['Healthy', 'Parkinson']))
	fig,ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_predict, y_test), colorbar=True, show_absolute=True, cmap='Blues')
	# ax.set_xticklabels([''] + labels)
	# ax.set_yticklabels([''] + labels)
	plt.savefig(os.path.join(XGB_results, str(i)+'_Testing_XGB.png'))
	plt.close()
    
	acc_test = round(clf.score(X_test,y_test)*100, 2)
	acc_train = round(clf.score(X_train,y_train)*100, 2)
	precision_test = round(precision_score(y_test, y_predict)*100, 2)
	precision_train = round(precision_score(y_train, y_predtr)*100, 2)
	recall_test = round(recall_score(y_test, y_predict)*100, 2)
	recall_train = round(recall_score(y_train, y_predtr)*100, 2)
	f1_test = round(f1_score(y_test,y_predict)*100, 2)
	f1_train = round(f1_score(y_train, y_predtr)*100, 2)

	return acc_test, acc_train, precision_test, precision_train, recall_test, recall_train, f1_test, f1_train, clf.best_params_



groups = df['name'].str[:-2] #Σβήνω από τη στήλη name, τους δύο τελευταίους χαρακτήρες, για να διαχωρίσω ποιες μετρήσεις 
							 #ανήκουν στο κάθε άνθρωπο. Κάθε μέτρησης φωνής στη στήλη name, έχει το εξής id: phon_R01_S01_1.
							 #Σβήνοντας τους τελευταίους δύο χαρακτήες σε όλα τα στοιχεία της στήλης name, προκύπτουν στοιχεία
							 #με ίδια id.

gss = GroupShuffleSplit(n_splits=10, train_size=.8, random_state=42) #Αρχικοποιώ το διαχωρισμό του συνόλου δεδομένων και να γίνει
																	 #10 φορές, με train_size = 0.8 και random_state = 42.
lpgo = LeavePGroupsOut(n_groups=1) #Αρχικοποιώ τη μεταβλητή lpgo, όπου θα με βοηθήσει στο GridSearchCV κάθε αλγορίθμου με στόχο να
								   #αφήσω εκτός 1 άτομο.

#print(len(list(np.unique(groups))))



#Σε κάθε αλγόριθμο, δημιουργώ λίστες για τα αποτελέσματα των μοντέλων τους, για περαιτέρω ανάλυση.
acc_NB_test = []
f1_NB_test = []
acc_NB_train = []
f1_NB_train = []
prec_NB_test = []
rec_NB_test = []
prec_NB_train = []
rec_NB_train = []
params_NB = []

acc_DT_test = []
f1_DT_test = []
acc_DT_train = []
f1_DT_train = []
prec_DT_test = []
rec_DT_test = []
prec_DT_train = []
rec_DT_train = []
params_DT = []

acc_Log_test = []
f1_Log_test = []
acc_Log_train = []
f1_Log_train = []
prec_Log_test = []
rec_Log_test = []
prec_Log_train = []
rec_Log_train = []
params_Log = []

acc_SV_test = []
f1_SV_test= []
acc_SV_train = []
f1_SV_train = []
prec_SV_test = []
rec_SV_test= []
prec_SV_train = []
rec_SV_train = []
params_SV = []

acc_RF_test = []
f1_RF_test = []
acc_RF_train = []
f1_RF_train = []
prec_RF_test = []
rec_RF_test = []
prec_RF_train = []
rec_RF_train = []
params_RF = []

acc_XGB_test = []
f1_XGB_test = []
acc_XGB_train = []
f1_XGB_train = []
prec_XGB_test = []
rec_XGB_test = []
prec_XGB_train = []
rec_XGB_train = []
params_XGB = []


#Στη συνέχεια με τη παρακάτω "for" προχωρώ στο training των αλγορίθμων, με 10 φορές spit του dataset και με τη μέθοδο LeaveOneGroupOut() sto GridSearchCV.
for i, (train_gss, test_gss) in enumerate(gss.split(X, y, groups)):
	#print('Ανθρωποι για training {} και για testing {} στον {}o διαχωρισμό του dataset'.format(int(len(list(y[train_gss]))/6), int(len(list(y[test_gss]))/6), count))
	print('Στον {}o διαχωρισμό του dataset έχουμε:'.format(i))


	acc_Gauss_test, acc_Gauss_train, precision_Gauss_test, precision_Gauss_train, recall_Gauss_test, recall_Gauss_train, f1_Gauss_test, f1_Gauss_train, params_Gauss = Gaussian(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo)
	acc_NB_test.append(acc_Gauss_test)
	acc_NB_train.append(acc_Gauss_train)
	f1_NB_test.append(f1_Gauss_test)
	f1_NB_train.append(f1_Gauss_train)
	prec_NB_test.append(precision_Gauss_test)
	prec_NB_train.append(precision_Gauss_train)
	rec_NB_test.append(recall_Gauss_test)
	rec_NB_train.append(recall_Gauss_train)
	params_NB.append(params_Gauss)

	acc_DecT_test, acc_DecT_train, precision_DecT_test, precision_DecT_train, recall_DecT_test, recall_DecT_train, f1_DecT_test, f1_DecT_train, params_DecT = DecisionTree(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo)
	acc_DT_test.append(acc_DecT_test)
	acc_DT_train.append(acc_DecT_train)
	f1_DT_test.append(f1_DecT_test)
	f1_DT_train.append(f1_DecT_train)
	prec_DT_test.append(precision_DecT_test)
	prec_DT_train.append(precision_DecT_train)
	rec_DT_test.append(recall_DecT_test)
	rec_DT_train.append(recall_DecT_train)
	params_DT.append(params_DecT)

	acc_LogReg_test, acc_LogReg_train, precision_LogReg_test, precision_LogReg_train, recall_LogReg_test, recall_LogReg_train, f1_LogReg_test, f1_LogReg_train, params_LogReg = LogReg(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo)
	acc_Log_test.append(acc_LogReg_test)
	acc_Log_train.append(acc_LogReg_train)
	prec_Log_test.append(precision_LogReg_test)
	prec_Log_train.append(precision_LogReg_train)
	rec_Log_test.append(recall_LogReg_test)
	rec_Log_train.append(recall_LogReg_train)
	f1_Log_test.append(f1_LogReg_test)
	f1_Log_train.append(f1_LogReg_train)
	params_Log.append(params_LogReg)
  
	acc_SVM_test, acc_SVM_train, precision_SVM_test, precision_SVM_train, recall_SVM_test, recall_SVM_train, f1_SVM_test, f1_SVM_train, params_SVM = SVM(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo) 
	acc_SV_test.append(acc_SVM_test)
	acc_SV_train.append(acc_SVM_train)
	prec_SV_test.append(precision_SVM_test)
	prec_SV_train.append(precision_SVM_train)
	rec_SV_test.append(recall_SVM_test)
	rec_SV_train.append(recall_SVM_train)
	f1_SV_test.append(f1_SVM_test)
	f1_SV_train.append(f1_SVM_train)
	params_SV.append(params_SVM)

	acc_XGBC_test, acc_XGBC_train, precision_XGBC_test, precision_XGBC_train, recall_XGBC_test, recall_XGBC_train, f1_XGBC_test, f1_XGBC_train, params_XGBC = XGBC(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo)   
	acc_XGB_test.append(acc_XGBC_test)
	acc_XGB_train.append(acc_XGBC_train)
	prec_XGB_test.append(precision_XGBC_test)
	prec_XGB_train.append(precision_XGBC_train)
	rec_XGB_test.append(recall_XGBC_test)
	rec_XGB_train.append(recall_XGBC_train)
	f1_XGB_test.append(f1_XGBC_test)
	f1_XGB_train.append(f1_XGBC_train)
	params_XGB.append(params_XGBC)

	acc_RFC_test, acc_RFC_train, precision_RFC_test, precision_RFC_train, recall_RFC_test, recall_RFC_train, f1_RFC_test, f1_RFC_train, params_RFC = RFC(X[train_gss], y[train_gss], X[test_gss], y[test_gss], groups[train_gss], lpgo)
	acc_RF_test.append(acc_RFC_test)
	acc_RF_train.append(acc_RFC_train)
	prec_RF_test.append(precision_RFC_test)
	prec_RF_train.append(precision_RFC_train)
	rec_RF_test.append(recall_RFC_test)
	rec_RF_train.append(recall_RFC_train)
	f1_RF_test.append(f1_RFC_test)
	f1_RF_train.append(f1_RFC_train)
	params_RF.append(params_RFC)



#Oι παρακάτω print, είναι για την διατύπωση των αποτελεσμάτων των αλγορίθμων.
print('Metrics on Testing set')
print('Naive-Bayes: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_NB_test), mean(prec_NB_test), mean(rec_NB_test), mean(f1_NB_test)))
print('Decision-Tree: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_DT_test), mean(prec_DT_test), mean(rec_DT_test), mean(f1_DT_test)))
print('Log-Reg: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_Log_test), mean(prec_Log_test), mean(rec_Log_test), mean(f1_Log_test)))
print('RFC: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_RF_test), mean(prec_RF_test), mean(rec_RF_test), mean(f1_RF_test)))
print('SVM: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_SV_test), mean(prec_SV_test), mean(rec_SV_test), mean(f1_SV_test)))
print('XGB: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_XGB_test), mean(prec_XGB_test), mean(rec_XGB_test), mean(f1_XGB_test)))
print('\n')

print('Metrics on Training set')
print('Naive-Bayes: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_NB_train), mean(prec_NB_train), mean(rec_NB_train), mean(f1_NB_train)))
print('Decision-Tree: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_DT_train), mean(prec_DT_train), mean(rec_DT_train), mean(f1_DT_train)))
print('Log-Reg: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_Log_train), mean(prec_Log_train), mean(rec_Log_train), mean(f1_Log_train)))
print('RFC: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_RF_train), mean(prec_RF_train), mean(rec_RF_train), mean(f1_RF_train)))
print('SVM: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_SV_train), mean(prec_SV_train), mean(rec_SV_train), mean(f1_SV_train)))
print('XGB: accuracy: {}, precision: {}, recall: {}, f1_score: {}'.format(mean(acc_XGB_train), mean(prec_XGB_train), mean(rec_XGB_train), mean(f1_XGB_train)))


#Δημιουργώ 2 dictionairies, για τη δημιουργία των αντίστοιχων τους DataFrames, για να κάνω τις συγκρίσεις μεταξύ των αλγορίθμων.
data_test = {'accuracy':[round(mean(acc_NB_test),2),round(mean(acc_DT_test),2),round(mean(acc_Log_test),2),
                         round(mean(acc_RF_test),2),round(mean(acc_SV_test),2),round(mean(acc_XGB_test),2)],
             'precision':[round(mean(prec_NB_test),2),round(mean(prec_DT_test),2),round(mean(prec_Log_test),2),
                          round(mean(prec_RF_test),2),round(mean(prec_SV_test),2),round(mean(prec_XGB_test),2)], 
            'recall':[round(mean(rec_NB_test),2),round(mean(rec_DT_test),2),round(mean(rec_Log_test),2),
                        round(mean(rec_RF_test),2),round(mean(rec_SV_test),2),round(mean(rec_XGB_test),2)],
            'f1_score':[round(mean(f1_NB_test),2),round(mean(f1_DT_test),2),round(mean(f1_Log_test),2),
                        round(mean(f1_RF_test),2),round(mean(f1_SV_test),2),round(mean(f1_XGB_test),2)]}

data_train = {'accuracy':[round(mean(acc_NB_train),2),round(mean(acc_DT_train),2),round(mean(acc_Log_train),2),
                          round(mean(acc_RF_train),2),round(mean(acc_SV_train),2),round(mean(acc_XGB_test),2)],
            'precision':[round(mean(prec_NB_train),2),round(mean(prec_DT_train),2),round(mean(prec_Log_train),2),
                          round(mean(prec_RF_train),2),round(mean(prec_SV_train),2),round(mean(prec_XGB_test),2)], 
            'recall':[round(mean(rec_NB_train),2),round(mean(rec_DT_train),2),round(mean(rec_Log_train),2),
                        round(mean(rec_RF_train),2),round(mean(rec_SV_train),2),round(mean(rec_XGB_train),2)],
            'f1_score':[round(mean(f1_NB_train),2),round(mean(f1_DT_train),2),round(mean(f1_Log_train),2),
                        round(mean(f1_RF_train),2),round(mean(f1_SV_train),2),round(mean(f1_XGB_train),2)]} 


df_train = pd.DataFrame(data_train, index =['Naive Bayes', 
                                            'Decision Tree',
                                            'Logistic Reg', 
                                            'Random Forest',
                                            'SVM',
                                            'XGB']) 

    
df_test = pd.DataFrame(data_test, index =['Naive Bayes', 
                                          'Decision Tree',
                                          'Logistic Reg', 
                                          'Random Forest',
                                          'SVM',
                                          'XGB']) 


# Δημιουργώ τα αντίστοιχα γραφήματα για σύγκριση των αλγορίθμων.
ax_train = df_train.plot(kind='bar', figsize=(16,6), rot=0, width=0.8)
plt.title('Training set')
plt.legend(bbox_to_anchor = (1,1))
plt.ylabel('Percentage (%)')
for p in ax_train.patches:
        ax_train.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.savefig(os.path.join(plots, 'Statistics_train.png'))
plt.close()
    
ax_test = df_test.plot(kind='bar', figsize=(16,6), rot=0, width=0.8)
plt.title('Testing set')
plt.legend(bbox_to_anchor = (1.12,1))
plt.ylabel('Percentage (%)')
for p in ax_test.patches:
        ax_test.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.savefig(os.path.join(plots, 'Statistics_test.png'))
plt.close()


#Στη συνέχεια βρίσκω ποιος αλγόριθμος έχει καλύτερη επίδπση και ποιος τη χειρότερη, στη μετρική F1-Score και στο train_set και στο test_set. 
print(df_train[df_train['f1_score']==df_train['f1_score'].max()])
print(df_train[df_train['f1_score']==df_train['f1_score'].min()])
print('\n')
print(df_test[df_test['f1_score']==df_test['f1_score'].max()])
print(df_test[df_test['f1_score']==df_test['f1_score'].min()])