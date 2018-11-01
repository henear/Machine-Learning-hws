import numpy as np
import matplotlib.pyplot as plt

a = np.zeros((25,4))
c = np.ones((25, 4))

b = np.arange(25)
# plt.plot(b, a[:,0],'white--',b,a[:,1]+1,'white--',b, a[:,2]+2,'white--',b,a[:,3]+3,'white--')



for i in range(25):
	for j in range(4):
		if c[i, j] == 1:
			plt.plot(i, j, marker= '.', markersize=3, color="red")
# plt.plot(b, a[:,0],'b--',b,a[:,1]+1,'b--',b, a[:,2]+2,'b--',b,a[:,3]+3,'b--')
x = [0,1]
y = [0,1]
plt.scatter(x,y)
plt.show()
# plt.grid(True)
# plt.show()