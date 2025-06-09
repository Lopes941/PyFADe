import pyfade as p
import numpy as np
from matplotlib import pyplot as plt

import numpy as np

x = np.array([1, 5, 9, 7,1,6,6,7])
y = np.array([4,2,3, 0,0,0,0,0])

z = x +1j * y 
Z = np.fft.fft(z)

print("FFT(x + i*y):", Z)



anom_signal = np.array([1,2,3,2,1])
SUBSEQ_SIZE = 2
a = p.mat_profile.Mat_Profile(anom_signal,SUBSEQ_SIZE)

a.set_cuda(False)
a.run_batch()

print(a.mp)
print(a.ind)