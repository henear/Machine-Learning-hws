import numpy as np  
import math  
import scipy.optimize as so 
import random
import matplotlib.pyplot as plt
import csv
import glob


def kernel(data1, data2, theta, wantderiv=True, measnoise=1.):
	
	theta = np.squeeze(theta)
	theta = np.exp(theta)
	if np.ndim(data1) == 1:
		d1 = np.shape(data1)[0]
		n = 1
	else:	
		(d1,n) = np.shape(data1)

	d2 = np.shape(data2)[0]
	sumxy = np.zeros((d1,d2))
	for d in range(n):
		D1 = np.transpose([data1[:,d]]) * np.ones((d1, d2))
		D2 = [data2[:,d]] * np.ones((d1,d2))
		sumxy += (D1 - D2) ** 2 * theta[d+1]

	k = theta[0] * np.exp(-0.5 * sumxy)

	if wantderiv:
		K = np.zeros((d1, d2, len(theta) + 1))
		K[:,:,0] = k + measnoise * theta[2] * np.eye(d1,d2)
		K[:,:,1] = k
		K[:,:,2] = -0.5 * k * sumxy
		K[:,:,3] = theta[2] * np.eye(d1,d2)
		return K
	else:
		return k + measnoise * theta[2] * np.eye(d1,d2)


def logPosterior(theta, data, t):
	k = kernel(data, data, theta, wantderiv=False)
	L = np.linalg.cholesky(k)
	beta = np.linalg.solve(L.transpose(), np.linalg.solve(L, t))
	logp = -0.5 * np.dot(t.transpose(), beta) - np.sum(np.log(np.diag(L))) - np.shape(data)[0] / 2. * np.log(2 * np.pi)
	return -logp 

def gradLogPosterior(theta, data, t):
	theta = np.squeeze(theta)
	d = len(theta)
	K = kernel(data, data, theta, wantderiv=True)

	L = np.linalg.cholesky(np.squeeze(K[:,:,0]))
	invk = np.linalg.solve(L.transpose(), 
			np.linalg.solve(L, np.eye(np.shape(data)[0])))

	dlogpdtheta = np.zeros(d)
	for d in range(1, len(theta) + 1):
		dlogpdtheta[d-1] = 0.5 * np.dot(t.transpose(), np.dot(invk, np.dot(np.squeeze(K[:,:,d]), np.dot(invk, t)))) - 0.5 * np.trace(np.dot(invk, np.squeeze(K[:,:,d])))
	return -dlogpdtheta

def plotGraph(data, t, theta, n, fingerX, start, step, timeVector, t2,wantCI):

	k = kernel(data, data, theta, wantderiv=False)
	L = np.linalg.cholesky(k)
	beta = np.linalg.solve(L.transpose(), np.linalg.solve(L, t))
	
	xStar = np.zeros((n*step,1))

	for i in range(n*step):
		xStar[i,0] = fingerX[i+start]

	kstar = kernel(data, xStar, theta, wantderiv= False,measnoise=0)
	f = np.dot(kstar.transpose(), beta)
	v = np.linalg.solve(L, kstar)
	V = kernel(xStar, xStar, theta, wantderiv=False, measnoise=0) - np.dot(v.transpose(), v)
	Var = list()
	lower = list()
	upper = list()
	if wantCI:
		for i in range(len(xStar)):
			currVar = V[i,i]
			upper.append(f[i] + 2.576 * math.sqrt(currVar))
			lower.append(f[i] - 2.576 * math.sqrt(currVar))
		p1 = plt.plot(t2, t, 'o')
		p2 = plt.plot(timeVector, f, 'r')
		p3 = plt.plot(timeVector, upper)
		p4 = plt.plot(timeVector, lower)
		plt.show()
	else: 
		return f


def processData(fingerX, targetX, theta, frameVector, start, numPts, step, wantCI):	
	print('Initial Guess sigmaf:', theta[0], 'sigmal:', theta[1], 'sigman:', theta[2])

	X = list()
	t = list()
	time = list()
	
	for i in range(numPts):
		curr = random.randint(0,9)
		X.append(fingerX[curr + i * step])
		t.append(targetX[curr + i * step])
		time.append(frameVector[curr + i * step])
	t2 = np.zeros((len(t), 1))
	X2 = np.zeros((len(X), 1))
	ti2 = np.zeros((len(X), 1))
	for i in range(len(t)):
		t2[i, 0] = t[i]
		X2[i, 0] = X[i]
		ti2[i, 0] = time[i]
	plotGraph(X2, t2, theta, numPts, fingerX, start, step, frameVector[start:start+numPts*step],ti2,wantCI)
	theta = np.copy(so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=(X2,t2), gtol=1e-4, disp=1))
	print(theta)	
	returns = plotGraph(X2, t2, theta, numPts, fingerX, start, step, frameVector[start:start+numPts*step],ti2, wantCI)
	if not wantCI:
		return returns
	# return theta


def getData():
	targetxs = list()
	frameVector = list()
	allFingerxs = list()
	# 5 and 20 below are use for hyper parameter tracing
	numPts = 102 # 5
	step = 10    # 20

	with open('./data_GP/YY/block1-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM/20161211143500-59968-right-speed_0.500.csv') as csvfile:
		reader = csv.DictReader(csvfile)		
		fingerxs = list()
		for row in reader:
			targetxs.append(float(row["target_x"]))
			fingerxs.append(float(row["finger_x"]))
			frameVector.append(float(row["frame"]))
	csvfile.close()
	inittheta = [1, 1, 0.1]

	start = 0
	# Trace the change of data to the change of hyper parameters
	# sigmaflist = list()
	# sigmallist = list()
	# sigmanlist = list()
	# indexlist = list()	
	# for i in range(100,610,500):
	# 	start = i
	# 	currTheta = processData(fingerxs, targetxs, inittheta, frameVector, start, numPts, step,wantCI = True)
	# 	sigmaflist.append(currTheta[0])
	# 	sigmallist.append(currTheta[1])
	# 	sigmanlist.append(currTheta[2])
	# 	indexlist.append(i)
	# plt.plot(indexlist,sigmaflist,'r',indexlist,sigmallist,'b',indexlist,sigmanlist,'g')
	# plt.show()
	
	processData(fingerxs, targetxs, inittheta, frameVector, start, numPts, step,wantCI = True)
	allFile = glob.glob("./data_GP/allData/*.csv")

	for fileDirec in allFile:
		with open(fileDirec) as singlecsv:
			singlereader = csv.DictReader(singlecsv)
			singleFingerxs = list()
			rowCount = 0
			for row in singlereader:
				singleFingerxs.append(float(row["finger_x"]))
			allFingerxs.append(singleFingerxs)
	
	allFingerxs = np.matrix(allFingerxs)
	
	# Tracing different object purpose ONLY
	# allThetasf = list()
	# allThetasl = list()
	# allThetasn = list()
	# curxaxis = list()
	# for i in range(60):
	# 	currFingerxs = list()
	# 	for j in range(1030):
	# 		currFingerxs.append(allFingerxs[i, j])
	# 	singleTheta = processData(currFingerxs, targetxs, inittheta, frameVector, start, numPts, step,wantCI = True)
	# 	allThetasf.append(singleTheta[0])
	# 	allThetasl.append(singleTheta[1])
	# 	allThetasn.append(singleTheta[2])
	# 	curxaxis.append(i)
	# plt.plot(curxaxis,allThetasf,'ro',curxaxis,allThetasl,'bo',curxaxis,allThetasn,'go')
	# plt.show()
	allFingerxs = allFingerxs.transpose()

	caringGroup = list()
	for i in range(1030):
		currIndex = random.randint(0, 59)

		caringGroup.append(allFingerxs[i,currIndex])
	returns = processData(caringGroup, targetxs, inittheta, frameVector, start, numPts,step, wantCI = False)
	plt.plot(frameVector[start:start+numPts*step], allFingerxs[start:start+numPts*step],'r',frameVector[start:start+numPts*step],returns,'b')
	plt.show()


if __name__ == '__main__':
	getData()
