import pyfade
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

c = pyfade.get_matrix_profile(np.ones(100),10,0,0)[:,0]


print(c)


plt.plot(c)
plt.show()

plt.grid(True)
