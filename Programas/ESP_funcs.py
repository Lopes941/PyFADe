import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from statsmodels.tsa.seasonal import MSTL, DecomposeResult
import stumpy
import re


def load_data():
    data_full = pd.read_csv("merged_data.csv",index_col='time')
    data_full.index = pd.to_datetime(data_full.index)
    data_full.index = data_full.index.tz_localize(None)
    data_full['Failure distance'] = pd.to_timedelta(data_full['Failure distance'])

    return data_full

def get_blocks(data, well, max_na = 24, min_size = 24, divider_val = 'VSD power frequency'):

    data_well = data[data['Well Run'] ==well]

    ind = data_well[divider_val].index

    # Getting first and last days
    year1 = np.min(ind.year)
    month1 = np.min(ind[ind.year==year1].month)

    year2 = np.max(ind.year)
    month2 = np.max(ind[ind.year==year2].month)+1
    if month2 == 13:
        month2 = 1
        year2 += 1

    # Creating date range
    first = datetime.datetime(year1, month1, 1, 0, 0, 0)
    last = datetime.datetime(year2, month2, 1, 0, 0, 0)
    dates = pd.date_range(start=first, end=last, freq='H')
    days = pd.DataFrame(index = dates, data = np.zeros(dates.size))
    days.loc[ind,0] = 1

    # Dividing by groups
    division_dates = ind[ind.to_series().diff().dt.days>max_na]
    blcks = [None]*(division_dates.size+1)
    nblcs = 0

    if division_dates.size == 0:
        blcks[nblcs] = ind
        nblcs = 1

    else:
        for i in range(division_dates.size):
            
            if i>0:
                sub_ind =  ind[np.logical_and(ind<division_dates[i],ind>=division_dates[i-1])]
            else:
                sub_ind = ind[ind<division_dates[i]]
        
            if sub_ind.size > 0: 
                if (sub_ind[-1]-sub_ind[0]).days > min_size:
                    blcks[nblcs] = sub_ind
                    nblcs += 1

        sub_ind =  ind[ind>=division_dates[i]]
        if sub_ind.size > 0:
            if (sub_ind[-1]-sub_ind[0]).days > min_size:
                blcks[nblcs] = sub_ind
                nblcs += 1


    return blcks

def remove_outliers(data, nIQR = 1.5, diff = 0, compensate = False):

    # Transforming to dataframe
    if not (isinstance(data,pd.DataFrame) or isinstance(data,pd.Series)):
        vals = data.copy()
        data = pd.Series(vals)

    # Getting diff data
    diff_data = data.copy()
    for i in range(diff):
        diff_data = diff_data.diff()

    # Detecting outliers
    if diff > 0:
        Q1 = diff_data.quantile(q=0.25)
        Q3 = diff_data.quantile(q=0.75)
    else:
        Q1 = diff_data.abs().quantile(q=0.25)
        Q3 = diff_data.abs().quantile(q=0.75)



    IQR = Q3 - Q1


    outliers = (Q1 - nIQR*IQR > diff_data.abs()) |  (diff_data.abs() > Q3 + nIQR*IQR)
    outliers = outliers + outliers.shift(-1).fillna(False)
    
    no_outliers = (outliers == False)

    # new_outliers = outliers.copy()
    # while new_outliers.size > 0:
    #     non_outlier_data = data[no_outliers].diff()
    #     new_outliers = (Q1 - nIQR*IQR > non_outlier_data.abs()) |  (non_outlier_data.abs() > Q3 + nIQR*IQR)
    #     new_outliers = new_outliers + new_outliers.shift(1).fillna(False)
    #     new_outliers = new_outliers[new_outliers].index
    #     outliers[new_outliers] = True
    #     no_outliers[new_outliers] = False
    
    if compensate and diff == 1:
        diff_data[no_outliers] = 0
        data -= diff_data.cumsum()

    data = data[no_outliers]

    return data, no_outliers

def fill_gaps(data, fill_type = 'spline', grace_after = 24, resamp = '1h', min_size = 1):

    # Resampling data
    resampled_data = data.resample(resamp).max()

    # Finding NAs (gaps)
    resampled_nas =  resampled_data[resampled_data.isnull()]
    resampled_nas[:] = resampled_nas.index
    resampled_nas = resampled_nas.shift(1)


    # Getting higher than one hour interval
    borders = resampled_nas[resampled_nas.index.diff()>datetime.timedelta(hours=1)]
    intervals = [None]*borders.size

    # Finding intervals
    grace_after = datetime.timedelta(hours=grace_after)
    min_size = datetime.timedelta(hours=min_size)
    k = 0
    for i in range(borders.size):
        if i == borders.size-1:
            start = borders.index[i]
            end =  resampled_nas.index[-1]
            size =  end - start
        else:
            start = borders.index[i]
            end =  borders.values[i+1]
            size = end-start

        #if k > 0:
         #   if start-
        if size >= min_size:
            intervals[k] = (start, size+grace_after)
            k += 1
    
    del intervals[k:]

    filled_data = data.resample(resamp).interpolate()
    filled_data = filled_data.bfill()
    filled_data = filled_data.ffill()
    filled_data = filled_data.fillna(0)


    return filled_data, intervals

def plot_recs(intervals,ax,data=None,ymax=0,ymin=0, add_pre=0):

    
    if data is not None:
        ymax = data.max()
        ymin = data.min()
    
    add_pre = datetime.timedelta(hours=add_pre)
    size = len(intervals)
    rect = [None]*size
    for i in range(size):
        rect[i] = Rectangle((intervals[i][0]-add_pre, ymin), intervals[i][1]+add_pre, (ymax-ymin), facecolor='lightgrey')
        ax.add_patch(rect[i])

def get_matrix_profile(data,m,intervals=None, quartile=None):

    index_start  = data.index[:-m+1]
    index_end  = data.index[m-1:]
    annotation_vector = np.ones(data.shape[0]-m+1)

    if intervals is not None:
        for start,size in intervals:
            end = start + size
            data[(data.index >= start) & (data.index <= end)] = 0
            ind = np.where((index_end >= start) & (index_start <= end))
            annotation_vector[ind] = 0

    values = data.values
    
    mp = stumpy.stump(values,m)[:,0]



    if intervals is not None:
        for start,size in intervals:
            end = start + size
            

        mp *= annotation_vector

    if quartile is not None:
        mp -= np.quantile(mp[annotation_vector.astype(bool)],quartile)
        mp[mp<0] = 0



    return mp