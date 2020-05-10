import numpy as np
import sklearn as sl
from sklearn import linear_model
import math


class MultiGaussClassify:

# Initializing
	def __init__(self,k,d):
		self.Mean = np.zeros((k,d))
		self.Sigma = np.identity(d)
		self.Prob = np.ones(k)*(1/k)
		self.prob = 0

	def fit(self,X,y,diag=False):
		Mean = np.mean(X,0)
		self.Sigma = np.matmul(np.transpose(X - Mean),X - Mean)/X.shape[0]
		if diag ==True:
			self.Sigma = self.Sigma*np.identity(X.shape[1])
		if np.linalg.det(self.Sigma)==0:
			self.Sigma = self.Sigma + 0.01*np.identity(X.shape[1])
		self.prob = math.log(np.linalg.det(self.Sigma))/2
# Computing array of classes
		self.classes = np.unique(y)

# Creating a dictionary, with class as keys and list of indices of row as values
		ci = {}
		for i,v in enumerate(y):
			try:
				ci[v].append(i)
			except KeyError:
				ci[v] = [i]

# Calculating mu, sigma and probability of each class in loop
		for q in range(self.classes.shape[0]):
			X_c = X[ci[self.classes[q]],:]
			self.Mean[q] = np.mean(X_c,0)
			self.Prob[q] = X_c.shape[0]/X.shape[0]
		return self

	def predict(self,X):
		y = np.zeros(X.shape[0]).astype(int)
# Finding Descriminant for each class and returning the index of maximum value and retuning the class from class array.
		for j in range(X.shape[0]):
			P = np.zeros(self.classes.shape[0])
			for i in range(self.classes.shape[0]):
				row = X[j]-self.Mean[i]
				row = np.array([row])
				P[i]= math.log(self.Prob[i]) - self.prob - np.matmul(np.matmul(row,np.linalg.inv(self.Sigma)),row.T)[0,0]/2
			max_prob = np.amax(P)
#finding indices of all maximum probabilities.
			ind = np.array([i for i,j in enumerate(P) if j==max_prob ])
			if len(ind)>1:
				np.random.shuffle(ind)
#Returning a random class from all with the max probability(if there are multiple)
			y[j] = self.classes[ind[0]]
		return y

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

#my_cross_val1 function for parameter overloading of fit for diagonal covariance
def my_cross_val1(method,X,y,k):
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
		model = method.fit(X_train,y_train,True)
		err[i] = ((np.sum((y_test != model.predict(X_test)).astype(int)))/len(y_test))
		e = err[i]
		print("Fold "+str(i+1) +": " + str(round(e*100,2))+"%")
	m = np.mean(err)
	s = np.std(err)
	print("Mean: "+str(round(m*100,2))+"%")
	print("Standard Deviation: "+str(round(s*100,2))+"%")
	return err



