import numpy as np
import random


col = 4
row = 3
numAward = 5
notReachVal = 0.8
def drawBoard():
	board = np.zeros((row, col))
	for i in range(numAward):
		awardVal = random.randint(1,5)
		c1 = random.randint(0, col-1)
		r1 = random.randint(0, row-1)
		while board[r1, c1] != 0:
			c1 = random.randint(0, col-1)
			r1 = random.randint(0, row-1)
		board[r1, c1] = random.randint(1,9)

	c2 = random.randint(0, col-1)
	while board[-1, c2] != 0:
		c2 = random.randint(0, col-1)
	board[-1, c2] = 10
	print(board)
	Rmatrix = CreateRMatrix(board)
	FinalQ = Qlearning(Rmatrix, board)
	print(FinalQ)

def Qlearning(Rmatrix, board):
	visited = np.zeros((row, col))
	QMatrix = np.zeros((row*col, row*col))
	numEpisode = 20
	learningRate = 0.8
	initrow = initcol = -1
	epi = 0
	
	# for epi in range(numEpisode):
	while epi < 10:
		initrow = random.randint(0,2)
		initcol = random.randint(0,3)
		visited[initrow, initcol] = 1
		QMatrixtemp = np.copy(QMatrix)

		while board[initrow, initcol]!=10:
			states = allPossibleState(initrow, initcol, visited, board)
			statelength = len(states)
			if statelength < 1:
				break
			currIndex = random.randint(0, statelength-1)
			nextState = board[states[currIndex][0], states[currIndex][1]]
			visited[states[currIndex][0],states[currIndex][1]] = 1
			nextStates = allPossibleState(states[currIndex][0],states[currIndex][1], visited, board)
			# Get max reward
			if len(nextStates) < 1:
				break
			currmax = 0
			for ii in nextStates:
				if QMatrixtemp[ii[0], ii[1]] > currmax:
					currmax = QMatrixtemp[ii[0], ii[1]]
			q1 = initrow*row + initcol
			q2 = states[currIndex][0]*row + states[currIndex][1]
			QMatrixtemp[q1, q2] = Rmatrix[q1, q2] + learningRate * currmax
			initrow = states[currIndex][0]
			initcol = states[currIndex][1]
		if board[initrow, initcol] == 10:
			QMatrix = np.copy(QMatrixtemp)
			epi += 1
			print("i")
	return QMatrix
			



def allPossibleState(initrow, initcol, visited, board):
	result = list()
	if initrow + 1 < row and 0 <= initcol < col and visited[initrow+1, initcol] == 0:
		temp = list()
		temp.append(initrow+1)
		temp.append(initcol)
		result.append(temp)
	if initrow - 1 >= 0 and 0 <= initcol < col and visited[initrow-1, initcol] == 0:
		temp = list()
		temp.append(initrow-1)
		temp.append(initcol)
		result.append(temp)
	if 0 <= initrow < row and initcol + 1 < col and visited[initrow, initcol+1] == 0:
		temp = list()
		temp.append(initrow)
		temp.append(initcol+1)
		result.append(temp)
	if 0 <= initrow < row and initcol - 1 >= 0 and visited[initrow, initcol-1] == 0:
		temp = list()
		temp.append(initrow)
		temp.append(initcol-1)
		result.append(temp)		
	return result




def CreateRMatrix(board):
	result = np.zeros((row*col, row*col))
	for i in range(row*col):
		currrow = i // col
		currcol = i % col
		
		if currrow < row - 1 and 0 < currcol < col - 1:
			for j in range(row*col):
				
				result[i, j] = -notReachVal
			result[i, i+1] = board[currrow, currcol+1]
			result[i, i-1] = board[currrow, currcol-1]
			
			result[i, i+col] = board[currrow+1, currcol]
		elif currrow == row-1:
			for j in range(row*col):
				result[i, j] = -notReachVal
			if currcol == 0:
				result[i, i+1] = board[currrow, currcol+1]
			elif 0 < currcol < col-1:
				result[i, i+1] = board[currrow, currcol+1]
				result[i, i-1] = board[currrow, currcol-1]
			else:
				result[i, i-1] = board[currrow, currcol-1]
		elif currcol == 0:
			for j in range(row*col):
				result[i, j] = -notReachVal
			result[i, i+1] = board[currrow, currcol+1]
			result[i, i+col] = board[currrow+1, currcol]
		else:
			for j in range(row*col):
				result[i, j] = -notReachVal
			result[i, i-1] = board[currrow, currcol-1]
			result[i, i+col] = board[currrow+1, currcol]

	print(result)
	return result

drawBoard()

			

