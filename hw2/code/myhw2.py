from scipy import io as sio
import scipy.io.wavfile as siw
import numpy as np
import math
from matplotlib import pyplot as plt
import numpy.linalg as la
import time
import sys


# For plot the small graph only
def plotGraph(icaA, icaU, icaX, smallTestresult):
	
	xaxis = np.arange(len(icaX[0]))
	print("haha" + str(len(xaxis)))
	plt.plot(xaxis, icaU[0], xaxis, icaU[1] + 1, xaxis, icaU[2] + 2)
	plt.show()
	plt.plot(xaxis, smallTestresult[0], xaxis, smallTestresult[1] +1, xaxis, smallTestresult[2] + 2)
	plt.show()

# Normalize every entry linearly, the min value corresponds to -1 and max to 1 
def normalize(smallTestresult):
	for i in range(len(smallTestresult)):
		currVector = smallTestresult[i]
		mini = min(smallTestresult[i])
		maxi = max(smallTestresult[i])
		slope = 2.0 / (maxi - mini)
		# slope * min + intercept = -0.5
		intercept = -1.0 + slope * maxi
		for j in range(len(smallTestresult[i])):
			smallTestresult[i][j] = slope * smallTestresult[i][j] + intercept

	return smallTestresult

# Calculate the norm 
def specs(a, b, c):
	return math.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2 + la.norm(c) ** 2)


# Calculate different combinations of sound
def errorCalculation(normsmallTestResult, icaU):
	test1 = normsmallTestResult[0]
	test2 = normsmallTestResult[1]
	test3 = normsmallTestResult[2]
	currmin = 1000;
	currmin = min(specs(test1 - icaU[0], test2 - icaU[1], test3 - icaU[2]), currmin)
	currmin = min(specs(test1 - icaU[0], test3 - icaU[1], test2 - icaU[2]), currmin)
	currmin = min(specs(test2 - icaU[0], test1 - icaU[1], test3 - icaU[2]), currmin)
	currmin = min(specs(test2 - icaU[0], test3 - icaU[1], test1 - icaU[2]), currmin)
	currmin = min(specs(test3 - icaU[0], test2 - icaU[1], test1 - icaU[2]), currmin)
	currmin = min(specs(test3 - icaU[0], test1 - icaU[1], test2 - icaU[2]), currmin)
	return currmin

# Load vectors in
def loadSound(yita, index):
	allSound = sio.loadmat("sounds.mat")

	# U is the matrix of size n * t
	icaData = sio.loadmat("icaTest.mat")

	realUtemp = allSound.get("sounds")
	icaA = icaData.get("A")
	icaU = icaData.get("U")

	# value 5 can be changed to 3 so that only takes 3 source sound from it and find what is the effect of recovery
	realU = np.zeros((5, len(realUtemp[0])))
	for i in range(5):
		for j in range(len(realUtemp[0])):
			realU[i, j] = realUtemp[i+index, j]
	n = len(realU)
	
	# Small test set
	# print("n is " + str(n) + "t is " + str(t))
	# icaX = np.dot(icaA, icaU)
	# smallTestresult, smallIter = gradientDescent(icaX, icaA, icaU, yita)
	# normsmallTestResult = normalize(smallTestresult)
	# plotGraph(icaA, icaU, icaX, normsmallTestResult)
	# error = errorCalculation(normsmallTestResult, icaU)
	# siw.write("small test.wav", 11025, normsmallTestResult)
	# Inspect result array element
	# xaxis = [0] * len(smallTestresult[0])


	# Now consider the real data set:
	realA = np.random.random((n, n))
	realX = np.dot(realA, realU)
	
	realResult, curIter = gradientDescent(realX, realA, realU, yita)
	
	realxaxis = np.arange(44000)
	error = 0
	# for i in range(3):
	# 	currVector = smallTestresult[i]
	# 	if la.norm(currVector - icaU[i]) > error:
	# 		error = la.norm(currVector - icaU[i])
	
	# For taking five source sound and recovery graph shown purpose
	# plt.plot(realxaxis, realX[0], realxaxis, realX[1] + 1, realxaxis, realX[2] + 2, realxaxis, realX[3] + 3, realxaxis, realX[4] + 4)
	realResult = normalize(realResult)
	plt.plot(realxaxis, realU[0], realxaxis, realU[1] + 1, realxaxis, realU[2] + 2, realxaxis, realU[3] + 3, realxaxis, realU[4] + 4)
	plt.show()
	plt.plot(realxaxis, realX[0], realxaxis, realX[1] + 1, realxaxis, realX[2] + 2, realxaxis, realX[3] + 3, realxaxis, realX[4] + 4)
	plt.show()
	plt.plot(realxaxis, realResult[0], realxaxis, realResult[1] +1, realxaxis, realResult[2] + 2, realxaxis, realResult[3]+3, realxaxis, realResult[4]+ 4)
	plt.show()

	# For taking three source sound and recovery graph shown purpose
	# plt.plot(realxaxis, realX[0], realxaxis, realX[1] + 1, realxaxis, realX[2] + 2)
	# realResult = normalize(realResult)
	# plt.plot(realxaxis, realU[0], realxaxis, realU[1] + 1, realxaxis, realU[2] + 2)
	# plt.show()
	# plt.plot(realxaxis, realX[0], realxaxis, realX[1] + 1, realxaxis, realX[2] + 2)
	# plt.show()
	# plt.plot(realxaxis, realResult[0], realxaxis, realResult[1] +1, realxaxis, realResult[2] + 2)
	# plt.show()
	# diff = np.subtract(realU, realResult)
	# relerror = 0
	# for i in range(5):
	# 	currVec = diff[i]
	# 	error = la.norm(currVec) / la.norm(realU[i])
	# 	relerror = max(relerror, error)
	# print(relerror)
	


# implement the algorithm
def gradientDescent(X, A, U, yita):
	m = len(A)
	n = len(A[0])
	W = 0.1 * np.random.random((n, m)) 
	
	curIter = 0
	numIter = 10000000
	Y = np.dot(W, X)
	for myIter in range(numIter):
		Y = np.dot(W, X)
		Z = np.zeros((len(Y), len(Y[0])))
		Z = 1.0 / (1.0 + np.exp(-Y))

		# deltaW = yita * (np.identity(n) + (1 - 2 * Z) * np.transpose(Y)) * W
		temp = (1 - 2*Z)
		
		YT = np.transpose(Y)
		temp2 = np.dot(temp,YT)
		
		Iden = np.identity(len(temp2))
		temp3 = np.add(Iden, temp2)
		deltaW = yita * np.dot(temp3, W)
		W = W + deltaW
		Y2 = np.dot(W, X)
		if la.norm(Y2 - Y) < 10 ** -7:
			print("myIter: " + str(myIter))
			break
		else:
			Y = np.copy(Y2)


	return np.dot(W, X), curIter



if __name__ == '__main__':
	# For tracking learning rate and error, number of iterations need to converge purpose ONLY
	# x = list()
	# y = list()

	# for i in np.arange(10**-4, 10**-2, 5*10**-4):
	# 	print(i)
	# 	curError = loadSound(i)
	# 	y.append(curError)
	# 	x.append(i)
	# plt.plot(x, y)
	# plt.show()
	# # plt.plot(x, z)
	# # plt.show()

	loadSound(10**-3, 0)
	#loadSound(10**-3, 1)
	#loadSound(10**-3, 2)


