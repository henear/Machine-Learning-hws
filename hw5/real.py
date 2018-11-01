import numpy as np
import random
import matplotlib.pyplot as plt
# R = np.array([[-1,-1,-1,-1,0,-1],[-1,-1,-1,0,-1,100],[-1,-1,-1,0,-1,-1],[-1,0,0,-1,0,-1],[0,-1,-1,0,-1,100],[-1,0,-1,-1,0,100]])
# Q = np.zeros(R.shape)
gamma = 0.8
row = 20
col = 5

# # Construct Q matrix
def createQMatrix(R):
	Q = np.zeros(R.shape)
	for epoch in range(3000):
		initialState = random.randint(0, R.shape[0]-1)
		poscand = list()
		for i in range(R.shape[0]):
			if R[initialState, i] != 0:
				poscand.append(i)

		curcandlength = len(poscand)
		randElement = random.randint(0, curcandlength-1)
		nextState = poscand[randElement]
		Q[initialState, nextState] = R[initialState, nextState] + gamma * max(Q[nextState,:])
		# if epoch % 100 == 0:
	# print(Q)
	return Q

def createLabel(board):
	a = 0
	labeldict = {}
	label = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			label[i, j] = a
			labeldict[a] = (i, j)
			a += 1

	return label, labeldict

def createSpec(board, flag, numSpecs):
	for numSpec in range(numSpecs):
		rcol = random.randint(0, col-1)
		rrow = random.randint(0, row-1)
		while board[rrow, rcol]!=0 or rrow == 0:
			rcol = random.randint(0, col-1)
			rrow = random.randint(0, row-1)
		
		board[rrow, rcol] = flag
		
	return board


def CreateRMatrix(board, labeldict):
	result = np.zeros((row*col, row*col))
	for i in range(row*col):
		currrow = labeldict[i][0]
		currcol = labeldict[i][1]
		for j in range(row*col):
			if j == i:
				result[i, i] = -10
		# deal with 4 corner case:
		if i == 0:
			result[0, 1] = board[0, 1]
			result[0, col] = board[1, 0]
		elif i == col-1:
			result[col-1, col-2] = board[0, -2]
			result[col-1, 2*col-1] = board[1, -1]
		elif i == col*(row-1):
			result[i, i+1] = board[-1, 1]
			result[i, i-col] = board[-2, 0]
		elif i == row * col - 1:
			result[i, i-1] = board[-1,-2]
			result[i, i-col] = board[-2, -1]
		elif 0 < currrow < row-1 and 0 < currcol < col - 1:
			# All 4 directions are possible
			result[i, i+1] = board[currrow, currcol+1]
			result[i, i-1] = board[currrow, currcol-1]
			result[i, i+col] = board[currrow+1, currcol]
			result[i, i-col] = board[currrow-1, currcol]
		# Deal with 4 side case
		elif currrow == 0:
			result[i, i+1] = board[currrow, currcol+1]
			result[i, i-1] = board[currrow, currcol-1]
			result[i, i+col] = board[currrow+1, currcol]
		elif currrow == row-1:
			result[i, i+1] = board[currrow, currcol+1]
			result[i, i-1] = board[currrow, currcol-1]
			result[i, i-col] = board[currrow-1, currcol]
		elif currcol == 0:
			result[i, i+col] = board[currrow+1, currcol]
			result[i, i-col] = board[currrow-1, currcol]
			result[i, i+1] = board[currrow, currcol+1]
		elif currcol == col-1:
			result[i, i+col] = board[currrow+1, currcol]
			result[i, i-col] = board[currrow-1, currcol]
			result[i, i-1] = board[currrow, currcol-1]


		

	# print(result)
	return result

def getMaxPossible(currlist, currrow, currcol, init, visited, Rlist):
	tempdict = {}
	ttlength = len(currlist)
	currmax = -20
	maxlab = 0
	if currcol - 1 >= 0 and visited[currrow, currcol-1] == 0 and Rlist[init-1] >= 0:
		tempdict[init-1] = currlist[init-1]
	if currcol + 1 < col and visited[currrow, currcol+1] == 0 and Rlist[init+1] >= 0:
		tempdict[init+1] = currlist[init+1]
	if currrow - 1 >= 0 and visited[currrow-1, currcol] == 0 and Rlist[init-col] >= 0:
		tempdict[init-col] = currlist[init-col]
	if currrow + 1 < row and visited[currrow+1, currcol] == 0 and Rlist[init+col] >= 0:
		tempdict[init+col] = currlist[init+col]
	for i in tempdict:
		if tempdict[i] >= currmax:
			currmax = tempdict[i]
			maxlab = i
	return maxlab





def getPath(R, Q, labeldict):
	init = random.randint(0, col-1)
	visited = np.zeros(np.shape(R))
	allpath = list()
	while init != R.shape[0]-1:
		# print(init)
		allpath.append(init)
		if len(allpath) > 100:
			break
		curmax = -1
		newinit = 0

		currrow = labeldict[init][0]
		currcol = labeldict[init][1]
		visited[currrow, currcol] = 1

		newinit = getMaxPossible(Q[init], currrow, currcol, init, visited, R[init])
		init = newinit
	allpath.append(init)
	# print(allpath)
	
	return allpath

def showResult(board, labeldict, allpath):
	for i in range(row):
		for j in range(col):
			if board[i, j] > 0:
				# plt.plot(i, j, marker= '.', markersize=5, color="red")
				pass
			if board[i, j] < 0:
				# plt.plot(i, j, marker= '.', markersize=5, color="m")
				pass
			if i == 0:
				# plt.plot(i, j, marker= '.', markersize=5, color="blue")
				pass
			if i == row - 1 and j == col -1:
				# plt.plot(i, j, marker= '.', markersize=5, color="black")
				pass
	# plt.plot(board)
	x = list()
	y = list()
	for i in range(len(allpath)):
		curPoint = allpath[i]
		x.append(labeldict[curPoint][0])
		y.append(labeldict[curPoint][1])


	# plt.grid(True)
	# plt.plot(x,y)
	# plt.show()

def getRewards(allpath, board, labeldict):
	reward = 0
	for n in allpath:
		i = labeldict[n][0]
		j = labeldict[n][1]
		if board[i, j] == 1:
			reward += 1
	return reward


def main(i):
	board = np.zeros((row, col))
	# Set the ending point
	board[-1, -1] = 10
	# Create reward
	board = np.copy(createSpec(board,1,i))
	# Create punishment
	board = np.copy(createSpec(board,-1,5))
	# label every point
	# print(board.T)
	label, labeldict = createLabel(board)
	
	# Create R matrix
	Rmatrix = CreateRMatrix(board, labeldict)
	Qmatrix = createQMatrix(Rmatrix)
	allpath = getPath(Rmatrix, Qmatrix, labeldict)
	while len(allpath) >= 100:
		allpath = getPath(Rmatrix, Qmatrix, labeldict)
	rewards = getRewards(allpath, board, labeldict)


	showResult(board, labeldict, allpath)
	return rewards
	
	

if __name__ == '__main__':
	x = list()
	y = list()

	for i in range(10,31,4):
		print(i)
		j = main(i)
		x.append(i)
		y.append(j*1.0/i)
	plt.plot(x,y)
	plt.show()