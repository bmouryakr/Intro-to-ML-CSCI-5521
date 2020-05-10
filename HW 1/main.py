import numpy as np
import sklearn as sl
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model

#my_cross_val function
def my_cross_val(method,X,y,k):
	# Split the datasets
	X_split = np.array_split(X,k)
	y_split = np.array_split(y,k)
	err = np.zeros(k)

# Assign Test and Train Set in loops
	for i in range(k):
		X_test = X_split[i]
		y_test = y_split[i]

		#counter to check if the test set is empty
		counter =0

		for j in range(k):
			if i == j:
				continue
			if counter ==0:
				X_train = X_split[j]
				y_train = y_split[j]
				counter = 1
				continue
			X_train = np.concatenate((X_train,X_split[j]))
			y_train = np.concatenate((y_train,y_split[j]))
		
		# Fit model and calculate error fraction 
		model = method.fit(X_train,y_train)
		err[i] = ((np.sum((y_test != model.predict(X_test)).astype(int)))/len(y_test))
		e = err[i]
		print("Fold "+str(i+1) +": " + str(round(e*100,2))+"%")
	m = np.mean(err)
	s = np.std(err)
	print("Mean: "+str(round(m*100,2))+"%")
	print("Standard Deviation: "+str(round(s*100,2))+"%")
	return err

#my_train_test function
def my_train_test(method,X,y,π,k):
	# Joining X and y
	d = np.concatenate((X,np.array([y]).T),axis = 1)
	err = np.zeros(k)

# Assign Test and Train Sets
	for i in range(k):
		#Randomizing the dataset
		np.random.shuffle(d)
		train = d[0:int((π*len(y))+1),:]
		test = d[int((π*len(y))+1):,:]
		
		
		# Fit model and calculate error fraction 
		model = method.fit(train[:,0:-1],train[:,-1].astype(int))
		err[i] = ((np.sum((test[:,-1].astype(int) != model.predict(test[:,0:-1])).astype(int)))/len(test[:,-1]))
		e = err[i]
		print("Fold "+str(i+1) +": " + str(round(e*100,2))+"%")
	m = np.mean(err)
	s = np.std(err)
	print("Mean: "+str(round(m*100,2))+"%")
	print("Standard Deviation: "+str(round(s*100,2))+"%")
	return err

#rand_proj function
def rand_proj(X,d):
	Q = np.random.randn(X.shape[1],d)
	X1 = np.matmul(X,Q)
	return X1

#quad_proj function
def quad_proj(X):
	d = X.shape[1]
	l = X.shape[0]
	#calculating the number of columns
	e = int((2*d)+(d*(d-1)/2))

	#initiating X2 with zeros 
	X2 = np.zeros((l,e))

	#Assigning X values
	X2[:,0:d] = X

	#Assigning X Squares
	X2[:,d:2*d] = np.square(X)

	start = 2*d

	#Assigning products
	for i in range(d-1):
		end = start+d-i-1
		X2[:,start:end] = np.multiply(X[:,i].reshape(l,1),X[:,(i+1):])
		start = end
	return X2














	