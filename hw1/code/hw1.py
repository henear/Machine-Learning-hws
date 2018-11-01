from scipy import io as sio
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math


def loadImage(numTrainImage, numTestImage, eigenValLarge, kValue):
	allData = sio.loadmat("digits.mat")
	trainlabels = allData.get("trainLabels")
	trainimages = allData.get("trainImages")
	testlabels = allData.get("testLabels")
	testimages = allData.get("testImages")

	column1 = len(trainimages)
	column2 = len(trainimages[0])

	allTrainLabel = np.zeros((1, numTrainImage))
	allTrainImage = np.zeros((column2*column1, numTrainImage))
	# load in train image data
	for imageLabelIter in range(numTrainImage):
		allTrainLabel[0, imageLabelIter] = trainlabels[0][imageLabelIter]

	for imageIter in range(numTrainImage):
		for myIter1 in range(column1):
			for myIter2 in range(column2):
				allTrainImage[myIter1* column1 + myIter2, imageIter] = trainimages[myIter2][myIter1][0][imageIter]
	# -1 means left over
	n 
	trainimages.reshape(28**2,)
	allTestImage = np.zeros((column2 * column1, numTestImage))
	allTestLabel = np.zeros((1, numTestImage))

	# load in test image data
	for testLabelIter in range(numTestImage):
		allTestLabel[0, testLabelIter] = testlabels[0][testLabelIter]

	# print(allTestLabel)
	for testImageIter in range(numTestImage):
		for myIter1 in range(column1):
			for myIter2 in range(column2):
				allTestImage[myIter1 * column1 + myIter2, testImageIter] = testimages[myIter2][myIter1][0][testImageIter]

	(mean, V) = hw1FindEigendigits(allTrainImage)
	# Show input test image
	# for i in range(200):
	# 	if i == 7:
	# 		plt.imshow(np.reshape(allTestImage[:,i], (28, 28)).transpose())
	# 		plt.show()

	# Only take care of first eigenvectors with large eigen values
	Vcare = V[:, 0:eigenValLarge]
	VTcare = np.transpose(Vcare)

	# map the training set to the smaller dimension
	trainImageNewBase = []
	for k in range(numTrainImage):
		currTrainImage = allTrainImage[:, k]
		
		diff = currTrainImage - mean
		newBaseVec = np.dot(VTcare, diff)
		trainImageNewBase.append(newBaseVec)
		
	trainImageNewBase = np.transpose(trainImageNewBase)

	testImage = []

	# map the test set to smaller dimension
	for k in range(numTestImage):
		currTestImage = allTestImage[:, k]
		
		diff = currTestImage - mean
		newBaseVec = np.dot(VTcare, diff)
		testImage.append(newBaseVec)
	testImage2 = np.transpose(testImage)

	totalResult = knn(kValue, testImage2, allTrainLabel, trainImageNewBase)
	correct = 0
	for i in range(len(totalResult)):
		if totalResult[i] == allTestLabel[0, i]:
			correct += 1
	accuracy = correct * 1.0 / len(totalResult)
	
	reconstructed = np.dot(testImage, VTcare)

	# for i in range(numTestImage):
	# 	currTestImagetemp = reconstructed[i]
	# 	currTestImage = currTestImagetemp + mean
	# 	if i  == 7:
	# 		plt.imshow(np.reshape(currTestImage, (28, 28)).transpose())

	# 		plt.show()
	
	return accuracy 
	
def knn(kValue, testImage, allTrainLabel, trainImageNewBase):

	testLength = len(testImage[0])
	trainLength = len(trainImageNewBase[0])
	
	totalResult = [0] * testLength
	for i in range(testLength):
		currTestPicture = testImage[:, i]
		record = list()
		# An array to record 
		tempresult = [0.0] * 10
		for j in range(trainLength):
			currTrainPicture = trainImageNewBase[:, j]
			# print(str(len(currTestPicture)) + " " + str(len(currTrainPicture)))
			currNorm = la.norm(currTrainPicture - currTestPicture)
			currList = list()
			currList.append((int)(allTrainLabel[0, j]))
			currList.append(currNorm)
			record.append(currList)
		record.sort(key=lambda x: (x[1]))
		# print(record)
		caredPoint = record[:kValue]
		maxCount = 0
		maxIter = 0

		# For each point, assign a different value, based on the distance, assign 
		# different weights to it

		for k in range(kValue):
			currPoint = record[k][0]
			tempresult[currPoint] += 1.0 / (k+1)

		for k2 in range(10):
			if tempresult[k2] > maxCount:
				maxCount = tempresult[k2]
				maxIter = k2
		totalResult[i] = maxIter
	return totalResult


def	hw1FindEigendigits(A):
	# print("in finding eigen digit")
	x = len(A)
	k = len(A[0])
	m = [0] * x
	
	tempSum = 0
	for i in range(x):
		tempSum = 0
		for j in range(k):
			tempSum += A[i][j]
		m[i] = tempSum * 1.0 / k
	
	V = np.zeros((x, k))
	for i in range(x):
		for j in range(k):
			V[i, j] = A[i, j] - m[i]
	
	VT = V.transpose()
	candidate = np.dot(VT, V)
	eigenVal, tempeigenVec = np.linalg.eig(candidate)
	eigenVec = np.dot(V, tempeigenVec)
	
	# Sort 
	# How shall we sort this?
	eigenVecValPair = list()
	for i in range(len(eigenVal)):
		currList = list()
		currList.append(eigenVal[i])
		currEigenvectors = list()
		
		for j in range(len(eigenVec)):
			currEigenvectors.append(eigenVec[j][i])
			
		currEigenvectors /= (1.0*la.norm(currEigenvectors))
		currList.append(currEigenvectors)
		
		eigenVecValPair.append(currList)

	eigenVecValPair.sort(key=lambda x: (-x[0]))
	V = np.zeros((x, k))
	
	for i in range(k):
		for j in range(x):
			V[j, i] = eigenVecValPair[i][1][j]

	# print("V specs")
	# Show eigenvectors
	# for i in range(len(V[0])):
	# 	plt.imshow(np.reshape(V[:,i], (28, 28)).transpose())
	# 	plt.show()
	return (m, V)

# test functions
if __name__ == '__main__':
	# the 4 parameters are numTrainImage, numTestImage, eigenValLarge, kValue respectively
	print(loadImage(1000, 450, 200, 20))




