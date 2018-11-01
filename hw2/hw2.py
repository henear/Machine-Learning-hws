from scipy import io as sio
import numpy as np
import math


def loadSound():
	allSound = sio.loadmat("sounds.mat")

	# U is the matrix of size n * t
	icaData = sio.loadmat("icaTest.mat")

	U = allSound.get("sounds")
	icaA = icaData.get("A")
	icaU = icaData.get("U")
	print("ica A data start")

	print(str(type(icaA)))
	print(len(icaA))
	print(len(icaA[0]))
	print("ica A data end")
	print("ica U data start")

	print(str(type(icaU)))
	print(len(icaU))
	print(len(icaU[0]))
	print("ica U data end")
	# print specs, for 
	# print(type(allSound2))
	# print(type(allSound2[0]))
	# print(type(allSound2[0][0]))
	# print(len(allSound2))
	# print(len(allSound2[0]))
	# print(len(allSound[0]))
	# print(type(allSound[0][0]))
	
	# n = len(U)
	# m = n*2
	# t = len(U[0])
	# A = np.random.random((m, n))
	# X = np.dot(A, U)
	# gradientDescent(X, A, U)
	# print("n is " + str(n) + "t is " + str(t))
	icaX = np.dot(icaA, icaU)
	gradientDescent(icaX, icaA, icaU)

# implement the algorithm
def gradientDescent(X, A, U):
	m = len(A)
	n = len(A[0])
	print("A specs")
	print("m is " + str(m))
	print("n is " + str(n))
	W = np.random.random((n, m))
	yita = 0.001 
	numIter = 2000
	for myIter in range(numIter):
		Y = np.dot(W, X)
		Z = np.zeros((len(Y), len(Y[0])))
		# learning rate
		if(myIter == 0):
			print("Z specs")
			print(len(Z))
			print(len(Z[0]))
	
	

		# for i in range(len(Z)):
		# 	for j in range(len(Z[0])):
		# 		Z[i, j] = 1.0  / (1.0 + math.exp(Y[i, j]))
		Z = 1.0 / (1.0 + np.exp(-Y))
		# deltaW = yita * (np.identity(n) + (1 - 2 * Z) * np.transpose(Y)) * W
		deltaW = np.dot((np.matrix(np.identity(n)) + (1 - 2*Z)*Y.transpose()), W)
		W = np.add(W, deltaW)

	return np.dot(W, X)




loadSound()