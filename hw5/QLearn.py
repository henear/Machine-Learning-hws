import Barrier
import numpy as np
import random
col = 4
row = 3
numBar = 1
numAward = numBar
notReachVal = -0.8



def drawBoard():
	board = np.zeros((row, col))
	# Setting start and end point
	board[0, 0] = 0.5
	board[-1, -1] = 0.6
	board = np.copy(assignBar(board))
	print(board)
	board = np.copy(assignReward(board))
	print(board)
	RMatrix = CreateRMatrix(board)
	print(board)
	QMatrix = np.zeros((row*col, row*col))
	QLearning(RMatrix, QMatrix)

# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
def QLearning(RMatrix, QMatrix):
	learningRate = 0.8
	epics = 10
	currcol = 0
	currrow = 0
	for epicIter in range(epics):		
		while currcol != col -1 or currrow != row-1:
			newCol, newRow = moveOneStep(currcol, currrow)
			possibles= {}
			if 0 <= newCol + 1 < col and 0 <= newRow < row:
				possibles[(newCol+1, newRow)] = RMatrix[newCol*col+newRow]
			currcol = newCol
			currrow = newRow
	pass
	


def moveOneStep(currcol, currrow):
	if currcol == 0 and currrow == 0:
		currdirec = random.randint(1,2)
		if currdirec == 1:
			currcol += 1
		else:
			currrow += 1
		return currcol, currrow
	elif currcol == 0 and currrow == row - 1:
		return currcol+1, currrow
	elif currcol == col - 1 and currrow == 0:
		currdirec = random.randint(1,2)
		if currdirec == 1:
			return currcol, currrow+1
		else:
			return currcol - 1, currrow

	elif 0 < currcol < col-1 and currrow < row -1:
		currdirec = random.randint(1,3)
		if currdirec == 1:
			return currcol+1, currrow
		elif currdirec == 2:
			return currcol, currrow+1
		else:
			return currcol-1, currrow
	elif currrow == row - 1:
		currdirec = random.randint(1, 2)
		if currdirec == 1:
			return currcol + 1, currrow
		else:
			return currcol - 1, currrow
	elif currcol == 0:
		currdirec = random.randint(1, 2)
		if currdirec == 1:
			return currcol, currrow + 1
		else:
			return currcol + 1, currrow
	else: # currcol is the last column
		currdirec = random.randint(1, 2)
		if currdirec == 1:
			return currcol, currrow - 1
		else:
			return currcol + 1, currrow




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



			
	for i in range(row*col):
		for j in range(row*col):
			if result[i, j] == 0.5 or result[i, j] == 0.6:
				result[i, j] = notReachVal

	print(result)
	return result




def assignBar(board):
	for i in range(numBar):
		currc = random.randint(0, col-1)
		currr = random.randint(0, row-1)
		while board[currr, currc] < 0:
			currc = random.randint(0, row-1)
			currr = random.randint(0, row-1)
		board[currr, currc] = -random.randint(1, 9)
	return board
		
	

def assignReward(board):
	for i in range(numAward):
		currc = random.randint(0, col-1)
		currr = random.randint(0, row-1)
		while board[currr, currc] != 0:
			currc = random.randint(0, row-1)
			currr = random.randint(0, row-1)
		board[currr, currc] = random.randint(1, 9)
	return board





drawBoard()	
