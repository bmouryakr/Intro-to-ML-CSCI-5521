import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import linear_model
from main import my_cross_val
from main import my_cross_val1
from main import MultiGaussClassify as MGC

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
methods = [MGC(2,Boston50.shape[1]-1),MGC(2,Boston75.shape[1]-1),MGC(np.unique(Digits).size,Digits.shape[1]-1),MGC(2,Boston50.shape[1]-1),MGC(2,Boston75.shape[1]-1),MGC(np.unique(Digits).size,Digits.shape[1]-1),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000)]
dat = [Boston50, Boston75, Digits,Boston50, Boston75, Digits,Boston50, Boston75, Digits]

num_folds = 5

method_name = ["MultiGaussClassify with full covariance matrix on Boston50","MultiGaussClassify with full covariance matrix on Boston75","MultiGaussClassify with full covariance matrix on Digits","MultiGaussClassify with diagonal covariance matrix on Boston50","MultiGaussClassify with diagonal covariance matrix on Boston75","MultiGaussClassify with diagonal covariance matrix on Digits","LogisticRegression with Boston50","LogisticRegression with Boston75","LogisticRegression with Digits"]

# calling my_cross_val in loops
for i in range(len(methods)):
	print(method_name[i]+": ")
	if i<3 or i>5:
		err = my_cross_val(methods[i],dat[i][:,0:-1],dat[i][:,-1],num_folds)
	else:
		err = my_cross_val1(methods[i],dat[i][:,0:-1],dat[i][:,-1],num_folds)

