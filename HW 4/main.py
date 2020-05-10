import numpy as np
import sklearn as sl
from sklearn import linear_model
import math


class MySVM2:

# Initializing
	def __init__(self,d,m,threshold):
		self.w = np.random.uniform(low=-0.01,high=0.01,size=(d+1,))
		self.m = m
		self.threshold = threshold

	def fit(self,X,y):
	#Padding Ones for X0
		X2 = np.concatenate((np.array([np.ones(X.shape[0])]).T,X),axis=1)
		zipp = np.concatenate((X2,np.array([y]).T),axis = 1)
	#Initializing Values
		tmax=15000 
		lamb = 0.00001
		self.w = np.random.uniform(low=-0.01,high=0.01,size=(X.shape[1]+1,))
		a=10000
	#Optimizing Weights
		for i in range(tmax):
			if self.m!='all':
				ind = np.random.choice(X.shape[0],self.m)
				X1=zipp[ind,0:-1]
				y2=zipp[ind,-1]
			else:
				X1 = X2
				y2 = y
	#Updating Values
			y1 = np.matmul(X1,self.w)
			y3 = np.multiply(y2,y1)
			y4=(y3<1).astype(int)
			dist = 1 - y3
			cost = sum(y4*dist)/X1.shape[0] + 2.5*(np.linalg.norm(self.w)**2)
			if a>cost:
				a=cost
				it = i
	#Limit for breaking loop
			if a <= self.threshold:
				break
			gradient = (np.matmul(X1.T,-y2*y4)/X1.shape[0])+5*self.w
			self.w = self.w - lamb*gradient
		#print("min cost "+str(a)+" at "+str(it))
		return self
	#Predicting
	def predict(self,X):
	#Padding Ones for X0
		X1 = np.concatenate((np.array([np.ones(X.shape[0])]).T,X),axis=1)
		y1 = np.matmul(X1,self.w)
		y1=(y1>0).astype(int)
		return (2*y1)-1




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
		pred = model.predict(X_test)
		err[i] = ((np.sum((y_test != pred).astype(int)))/len(y_test))
		e = err[i]
		print("Fold "+str(i+1) +": " + str(round(e*100,2))+"%")
		print
	m = np.mean(err)
	s = np.std(err)
	print("Mean: "+str(round(m*100,2))+"%")
	print("Standard Deviation: "+str(round(s*100,2))+"%")
	return err




