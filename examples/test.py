import pyfade as p
import numpy as np
from matplotlib import pyplot as plt


# series = np.array([1,2,5,7,1,2,6,7])
N = 1<<10

x = np.linspace(0,100,N)
series = np.sin(x)

a = p.mat_profile.Mat_Profile(series,30)

a.run_batch()

plt.plot(a.series)
plt.figure()
plt.plot(a.mp)

plt.show()