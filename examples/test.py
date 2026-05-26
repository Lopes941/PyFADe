import numpy as np
import matplotlib.pyplot as plt

import pyfade
from pyfade.series import DataFrameBuilder, Interpolation, Extrapolation
from pyfade.window import WindowBuilder
from pyfade.group import FeatureGroups, FeatureGroupBuilder
from pyfade.features import Features

data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
builder = DataFrameBuilder(data)
dataframe = builder.set_interpolation(Interpolation.LINEAR).set_extrapolation(Extrapolation.CONSTANT).build()

window_builder = WindowBuilder(dataframe)
window = window_builder.set_window_size(2).build()

group_builder = FeatureGroupBuilder(FeatureGroups.ContinuousStatistics, window)
group_builder.build()

fig, axs = window.plot_feature(pyfade.features.Features.Mean)


axs[0].grid(True)
plt.show()

exit()

WINDOW_SIZE = 11
def example_dataset(data:np.ndarray):
    builder = pyfade.series.DataFrameBuilder(data)
    dataset = builder.build()

    # Creating the window
    window_builder = pyfade.window.WindowBuilder(dataset)
    window_builder.set_window_size(WINDOW_SIZE)
    window = window_builder.build()

    # Adding the feature
    feature_builder = pyfade.FeatureGroupBuilder(pyfade.FeatureGroups.ContinuousStatistics,window)
    feature_builder.build()

    group_builder = pyfade.FeatureGroupBuilder(pyfade.FeatureGroups.MatrixProfile, window)
    group_builder.set_parameter('use_cuda',False)\
                .set_parameter('left_only',True)\
                .set_parameter('skip_start',0)\
                .set_parameter('quantile_threshold',0)
    group_builder.build()

    return dataset, window


N = 50
n = np.arange(N,dtype=float)

y = np.sin(n*0.5)

# Defining the dataset object
dataset, window = example_dataset(y)
a = window.mp.copy()
plt.plot(window.mp.T);
plt.title("Batch study")


# Defining the dataset object
start = 21
update_size = 11
dataset, window = example_dataset(y[:start])

plt.figure()
plt.plot(window.mp.T);
s = window.mp.size

plt.title("Real-time study")

style = ['g-','r-','y-','g-','r-','y-','g-','r-','y-']

end = start + update_size
k = 0
while end < y.size:
    print(start,s)
    dataset.insert_data(y[start:end])
    plt.plot(np.arange(s,window.mp.size),window.mp[:,s:].T,style[k]);
    s = window.mp.size
    
    start = end
    end = start + update_size
    k += 1
end = y.size
dataset.insert_data(y[start:end])
plt.plot(np.arange(s,window.mp.size),window.mp[:,s:].T,style[k]);
# plt.figure()
# plt.plot(window.mp.T);
# plt.title("Real-time study")
plt.grid(True)
plt.figure()
plt.plot((a-window.mp).T)
plt.grid()
plt.show()