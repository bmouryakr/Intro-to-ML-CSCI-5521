import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import linear_model
from main import my_cross_val
from main import MyLogisticReg2 as MLR2

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

# defining a list of methods and data
methods = [MLR2(Boston50.shape[1]-1),sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000)]
dat = [Boston50, Boston75]

num_folds = 5

dat_name = {0 : "Boston50", 1 : "Boston75"}
method_name = {0 : "MyLogisticReg2", 1 : "LogisticRegression"}

old_settings=np.seterr(all='ignore')

# calling my_cross_val in loops
for i in range(len(methods)):
	for j in range(len(dat)):
		print("Error rates for "+ method_name[i] + " with " + dat_name[j]+": ")
		err = my_cross_val(methods[i],dat[j][:,0:-1],dat[j][:,-1],num_folds)

