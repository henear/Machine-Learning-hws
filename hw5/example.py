import numpy as np
import random

R = np.array([[-1,-1,-1,-1,0,-1],[-1,-1,-1,0,-1,100],[-1,-1,-1,0,-1,-1],[-1,0,0,-1,0,-1],[0,-1,-1,0,-1,100],[-1,0,-1,-1,0,100]])
Q = np.zeros(R.shape)
gamma = 0.8
print(R)
print(Q)
initialState = 1
poscand = list()
for i in range(6):
	if R[initialState, i] != -1:
		poscand.append(i)
print(poscand)
nextState = poscand[-1]
Qtemp = np.copy(Q)
# for i in range(6):
# if R[i, nextState]!=-1:
Qtemp[initialState, nextState] = R[initialState, nextState] + gamma *  max(max(Qtemp[nextState,:]),0)
Q = np.copy(Qtemp)


initialState = 3
nextState = 1
Q[initialState, nextState] = R[initialState, nextState] + gamma * max(Q[nextState,:])
print(Q)
print(nextState)
initialState = 1
nextState = 5
Q[initialState, nextState] = R[initialState, nextState] + gamma * max(Q[nextState,:])
print(Q)
print(nextState)

