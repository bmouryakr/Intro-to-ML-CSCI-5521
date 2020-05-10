import numpy as np
import sklearn as sl
from sklearn import linear_model
import math


class MyLogisticReg2:

# Initializing
	def __init__(self,d):
		self.w = np.random.uniform(low=-0.01,high=0.01,size=(d+1,))

#Defining the Sigmoid function
	def sigmoid(self,x):
		return 1/(1+np.exp(-x))


	def fit(self,X,y):
	#Padding Ones for X0
		X1 = np.concatenate((np.array([np.ones(X.shape[0])]).T,X),axis=1)
	#Initializing Values
		tmax=150000
		lamb = 0.3
		self.w = np.random.uniform(low=-0.01,high=0.01,size=(X.shape[1]+1,))
		#self.w = 0*self.w
		previous = 5*np.ones(X.shape[1]+1)
		m=1000
	#Optimizing Weights
		for i in range(tmax):
			d = np.linalg.norm(previous-self.w)
			previous = self.w
			if m>d:
				m=d
	#Limit for breaking loop
			if m<=1.5:
				break
	#Updating Values
			y1 = np.matmul(X1,self.w)
			y1 = self.sigmoid(y1)
			gradient = np.matmul(X1.T,(y1-y))/X.shape[0]
			self.w = self.w - lamb*gradient
		return self
	#Predicting
	def predict(self,X):
		X1 = np.concatenate((np.array([np.ones(X.shape[0])]).T,X),axis=1)
		y1 = np.matmul(X1,self.w)
		y1 = self.sigmoid(y1)
		y1=(y1>=0.5).astype(int)
		return y1



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




