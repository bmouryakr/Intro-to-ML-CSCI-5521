import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from main import rand_proj
from main import quad_proj
from main import my_cross_val

#loading digits dataset
digits,dresp = sl.datasets.load_digits(return_X_y=True)


#Joining X and Y to shuffle
Digits = np.concatenate((digits,np.array([dresp]).T),axis = 1)

np.random.shuffle(Digits)

dim = 32

#Calling rand_proj and quad_proj functions
X1 = rand_proj(Digits[:,0:-1],dim)
X2 = quad_proj(Digits[:,0:-1])

# defining a list of methods and data
methods = [sl.svm.LinearSVC(max_iter=2000),sl.svm.SVC(gamma='scale',C=10),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000)]
dat_X = [X1,X2]
dat_y = Digits[:,-1]

num_folds = 10

dat_name = {0 : u"X\u0303\u2081", 1 : u"X\u0303\u2082"}
method_name = {0 : "LinearSVC", 1 : "SVC", 2 : "LogisticRegression"}

# calling my_cross_val in loops
for i in range(len(methods)):
	for j in range(len(dat_X)):
		print("Error rates for "+ method_name[i] + " with " + dat_name[j]+": ")
		err = my_cross_val(methods[i],dat_X[j],dat_y,num_folds)