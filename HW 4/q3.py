import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import linear_model
from main import my_cross_val
from main import MySVM2

# loading boston dataset
boston,resp = sl.datasets.load_boston(return_X_y = True)

#Joining X and Y to sort and shuffle
boston = np.concatenate((boston,np.array([resp]).T),axis = 1)

#Creating Boston50 by sorting based on response and reassigning response values
Boston50 = boston[np.argsort(boston[:,-1])]
Boston50[0:int((boston.shape[0])/2),-1] = -1
Boston50[int((boston.shape[0])/2):,-1] = 1

# randomising the dataset
np.random.shuffle(Boston50)


#Creating boston75 by sorting based on response and reassigning response values
Boston75 = boston[np.argsort(boston[:,-1])]
Boston75[0:int(3*(boston.shape[0])/4),-1] = -1
Boston75[int(3*(boston.shape[0])/4):,-1] = 1

# randomising the dataset
np.random.shuffle(Boston75)

# defining a list of methods and data
methods = [ MySVM2(Boston50.shape[1]-1,40,0.12),#12
			MySVM2(Boston50.shape[1]-1,200,0.30),#30
			MySVM2(Boston50.shape[1]-1,'all',0.45),#45
			sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000),
			MySVM2(Boston75.shape[1]-1,40,0.08),#08
			MySVM2(Boston75.shape[1]-1,200,0.25),#25
			MySVM2(Boston75.shape[1]-1,'all',0.4),#40
			sl.linear_model.LogisticRegression(penalty='l2',solver='lbfgs',multi_class='multinomial',max_iter=5000)
			]
dat = [Boston50, Boston75]

num_folds = 5

method_name = {  0 : "MySVM2 with m = 40 for Boston50",
				 1 : "MySVM2 with m = 200 for Boston50", 
				 2 : "MySVM2 with m = n for Boston50",
				 3 : "LogisticRegression for Boston50",
				 4 : "MySVM2 with m = 40 for Boston75",
				 5 : "MySVM2 with m = 200 for Boston75",
				 6 : "MySVM2 with m = n for Boston75",
				 7 : "LogisticRegression for Boston75"}

old_settings=np.seterr(all='ignore')

# calling my_cross_val in loops
for i in range(len(methods)):
		print("Error rates for "+ method_name[i] )
		err = my_cross_val(methods[i],dat[int(i/4)][:,0:-1],dat[int(i/4)][:,-1],num_folds)

