import pyfade
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt



c = pyfade.get_matrix_profile(a,10,'both')[:,0]


print(c)


plt.plot(c)
plt.show()

plt.grid(True)
