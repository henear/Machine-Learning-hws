import numpy as np
import random

R = np.array([[-1,-1,-1,-1,0,-1],[-1,-1,-1,0,-1,100],[-1,-1,-1,0,-1,-1],[-1,0,0,-1,0,-1],[0,-1,-1,0,-1,100],[-1,0,-1,-1,0,100]])
Q = np.zeros(R.shape)
gamma = 0.8

# Construct Q matrix
for epoch in range(3000):
	initialState = random.randint(0, R.shape[0]-1)
	poscand = list()
	for i in range(R.shape[0]):
		if R[initialState, i] != -1:
			poscand.append(i)

	curcandlength = len(poscand)
	randElement = random.randint(0, curcandlength-1)
	nextState = poscand[randElement]
	Q[initialState, nextState] = R[initialState, nextState] + gamma * max(Q[nextState,:])

init = 2
rewards = 0

# print(Q[init,:]

while init != R.shape[0]-1:
	print(init)
	curmax = -1
	newinit = 0
	for i in range(R.shape[0]):
		if Q[init, i] > curmax:
			newinit = i
			curmax = Q[init, i]
	rewards += R[init, newinit]
	init = newinit
print(rewards)
print(init)

