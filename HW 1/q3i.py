import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from main import my_cross_val

# loading boston dataset
boston,resp = sl.datasets.load_boston(return_X_y = True)

#Joining X and Y to sort and shuffle
boston = np.concatenate((boston,np.array([resp]).T),axis = 1)

#Creating Boston50 by sorting based on response and reassigning response values
Boston50 = boston[np.argsort(boston[:,-1])]
Boston50[0:int((boston.shape[0])/2),-1] = 0
Boston50[int((boston.shape[0])/2):,-1] = 1

# randomising the dataset
np.random.shuffle(Boston50)


#Creating boston75 by sorting based on response and reassigning response values
Boston75 = boston[np.argsort(boston[:,-1])]
Boston75[0:int(3*(boston.shape[0])/4),-1] = 0
Boston75[int(3*(boston.shape[0])/4):,-1] = 1

# randomising the dataset
np.random.shuffle(Boston75)


# loading Digits data
digits,dresp = sl.datasets.load_digits(return_X_y=True)


#Joining X and Y to shuffle
Digits = np.concatenate((digits,np.array([dresp]).T),axis = 1)

np.random.shuffle(Digits)


# defining a list of methods and data
methods = [sl.svm.LinearSVC(max_iter=2000),sl.svm.SVC(gamma='scale',C=10),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000)]
dat = [Boston50, Boston75, Digits]

num_folds = 10

dat_name = {0 : "Boston50", 1 : "Boston75", 2 : "Digits"}
method_name = {0 : "LinearSVC", 1 : "SVC", 2 : "LogisticRegression"}

# calling my_cross_val in loops
for i in range(len(methods)):
	for j in range(len(dat)):
		print("Error rates for "+ method_name[i] + " with " + dat_name[j]+": ")
		err = my_cross_val(methods[i],dat[j][:,0:-1],dat[j][:,-1],num_folds)
