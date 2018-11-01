from scipy import io as sio
import numpy as np
allData = sio.loadmat("digits.mat")
print(type(allData.get("trainLabels")))

