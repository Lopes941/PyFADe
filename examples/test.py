import pyfade as p
import numpy as np
from matplotlib import pyplot as plt


anom_signal = np.array([1,2,3,2,1])
SUBSEQ_SIZE = 2
a = p.mat_profile.Mat_Profile(anom_signal,SUBSEQ_SIZE)

a.run_batch()

print(a.mp)
print(a.ind)