
import numpy as np
import pandas
import pywt

import matplotlib.pyplot as plt

from ..core import IFeatureGroup
from ..core import IFeatureGroupParameter
from ..core import UseCudaParam, LeftOnlyParam, SkipStartParam, ExclusionZoneRatioParam, QuantileParam

from ..features import Features
from ..series.series import DataFrame

HSPACE = 0.2
class Decomposition:

    def _decompose_signal(self,data: np.ndarray, index):

        # Creating variables
        self.decomp = [None]*self.level
        cA = data.copy()

        for lvl in range(self.level):
            #Performing discrete wavelet transform
            cA, cD = pywt.dwt(cA.copy(), wavelet=self.wavelet)

            coef_A = [cA, None] + [None]*lvl
            coef_D = [None, cD] + [None]*lvl

            # Reconstructing both signals
            sig_A = pywt.waverec(coef_A, wavelet=self.wavelet)
            sig_D = pywt.waverec(coef_D, wavelet=self.wavelet)

            new_ind_A = pandas.date_range(start=index.min(), end=index.max(), periods=sig_A.size)
            new_ind_D = pandas.date_range(start=index.min(), end=index.max(), periods=sig_D.size)

            sig_A = np.interp(index.astype(int),new_ind_A.values.astype(int),sig_A)
            sig_D = np.interp(index.astype(int),new_ind_D.values.astype(int),sig_D)

            #Writing decomposed signals
            self.decomp[lvl] = [pandas.Series(sig_A,index=index), pandas.Series(sig_D,index=index)]
            self.decomp[lvl][0].name = f'Approximation {lvl+1}'
            self.decomp[lvl][1].name = f'Detail {lvl+1}'

    def __init__(self, data: np.ndarray, index, wavelet: str, level: int):

        self.wavelet = wavelet
        self.level = level
        self._decompose_signal(data,index)

    def get_dataframe_from_decomp(self):
        pass

    
    def plot(self,\
            height: float = 3,\
            width: float = 10,):

        fig, axs = plt.subplots(self.level, 2, sharex=True, gridspec_kw={'hspace': HSPACE},figsize=[width,height*self.level])

        for lvl in range(self.level):

            #Approximation
            ax = axs[lvl,0]
            ind = self.decomp[lvl][0].index
            sig = self.decomp[lvl][0].values
            ax.plot(ind,sig,'-b',markersize=2,linewidth=1)
            ax.grid(True)
            ax.set_ylabel(f'Approximation {lvl+1}')


            #Approximation
            ax = axs[lvl,1]
            ind = self.decomp[lvl][1].index
            sig = self.decomp[lvl][1].values
            ax.plot(ind,sig,'-b',markersize=2,linewidth=1)
            ax.grid(True)
            ax.set_ylabel(f'Detail {lvl+1}')

        return fig,axs


class TotalWaveletLevel(IFeatureGroupParameter):

    @staticmethod
    def static_name():
        return 'wavelet_total_level'
    
    def __init__(self, val: int | float | bool = 3):
        super().__init__()
        self.set(val)

    def name(self):
        return TotalWaveletLevel.static_name()

    def set(self, value):
        self.value = value

class AggregateWaveletLevel(IFeatureGroupParameter):

    @staticmethod
    def static_name():
        return 'wavelet_aggregate_level'
    
    def __init__(self, val: int | float | bool = 1):
        super().__init__()
        self.set(val)

    def name(self):
        return AggregateWaveletLevel.static_name()

    def set(self, value):
        self.value = value

class WaveletType(IFeatureGroupParameter):

    @staticmethod
    def static_name():
        return 'wavelet_type'
    
    def __init__(self, val: int | float | bool = 'haar'):
        super().__init__()
        self.set(val)

    def name(self):
        return WaveletType.static_name()

    def set(self, value):
        self.value = value


class WaveletProfile(IFeatureGroup):

    @staticmethod
    def static_name():
        return 'wavelet_profile'
    
    @staticmethod
    def static_feature_names():
        return [Features.WaveletMP.value,
                'decomposition']
    
    @staticmethod
    def static_requirements():
        return []
    
    @staticmethod
    def static_parameters():
        return [TotalWaveletLevel.static_name(),
                UseCudaParam.static_name(), 
                LeftOnlyParam.static_name(), 
                SkipStartParam.static_name(), 
                ExclusionZoneRatioParam.static_name(), 
                QuantileParam.static_name()
                ]
    
    def __init__(self, dataset: DataFrame, requirements: list[IFeatureGroup]):
        super().__init__()

        from ..features.decomposition import WaveletMP
        self.wave_mp: WaveletMP = WaveletMP()

        self.total_level: TotalWaveletLevel = TotalWaveletLevel()
        self.aggregate_level: AggregateWaveletLevel = AggregateWaveletLevel()
        self.wavelet_type: WaveletType = WaveletType()

        self.use_cuda: UseCudaParam = UseCudaParam()
        self.left_only: LeftOnlyParam = LeftOnlyParam()
        self.skip_start: SkipStartParam = SkipStartParam()
        self.exclusion_zone: ExclusionZoneRatioParam = ExclusionZoneRatioParam()
        self.quantile: QuantileParam = QuantileParam()
        

    def name(self):
        return WaveletProfile.static_name()
    
    def feature_names(self):
        return WaveletProfile.static_feature_names()
    
    def requirements(self):
        return WaveletProfile.static_requirements()
    
    def parameters(self):
        return WaveletProfile.static_parameters()
    
    def set_parameter(self, parameter_name: str, parameter_value: int | float | bool):
        if parameter_name == TotalWaveletLevel.static_name():
            self.total_level.set(parameter_value)
        elif parameter_name == AggregateWaveletLevel.static_name():
            self.aggregate_level.set(parameter_value)
        elif parameter_name == WaveletType.static_name():
            self.wavelet_type.set(parameter_value)

        elif parameter_name == UseCudaParam.static_name():
            self.use_cuda.set(parameter_value)
        elif parameter_name == LeftOnlyParam.static_name():
            self.left_only.set(parameter_value)
        elif parameter_name == SkipStartParam.static_name():
            self.skip_start.set(parameter_value)
        elif parameter_name == ExclusionZoneRatioParam.static_name():
            self.exclusion_zone.set(parameter_value)
        elif parameter_name == QuantileParam.static_name():
            self.quantile.set(parameter_value)

    
    def get_feature(self, name) -> np.ndarray:
        if name == Features.WaveletMP.value:
            return self.wave_mp.feature
        elif name == 'decomposition':
            return self.decomposition
    
    def update(self, dataset, window_size: int):

        from ..series import DataFrameBuilder
        from ..window import WindowBuilder, FeatureGroupBuilder
        
        self.decomposition: list = [None]*dataset.ndim
        mp_params = {
            'use_cuda': self.use_cuda.get(),
            'left_only': self.left_only.get(),
            'skip_start': self.skip_start.get(),
            'quantile_threshold': self.quantile.get(),
            'exclusion_zone_ratio': self.exclusion_zone.get()
        }

        # Decomposing signals
        for i in range(dataset.ndim):
            self.decomposition[i] = Decomposition(dataset.data[i,:],dataset.index,self.wavelet_type.value, self.total_level.value)

        # Getting MPs from each signal
        self.wave_mp.feature = np.zeros((dataset.ndim, dataset.len-window_size+1))
        for i in range(dataset.ndim):

            # Creating dataframe for each value
            data = np.zeros((self.total_level.value+1, dataset.len))
            for lvl in range(self.total_level.value):
                data[lvl,:] = self.decomposition[i].decomp[lvl][1].values
            data[-1,:] = self.decomposition[i].decomp[-1][0].values

            frame = DataFrameBuilder(data).build()
            window = WindowBuilder(frame).set_window_size(window_size).build()
            from ..group import FeatureGroups
            FeatureGroupBuilder(FeatureGroups.MatrixProfile,window).set_parameters(mp_params).build()
            FeatureGroupBuilder(FeatureGroups.KProfile,window).build()

            self.wave_mp.feature[i,:] = window.kp[self.aggregate_level.value,:]


class WaveletKProfile(IFeatureGroup):

    @staticmethod
    def static_name():
        return 'wavelet_k_profile'
    
    @staticmethod
    def static_feature_names():
        return [Features.WaveletKP.value]
    
    @staticmethod
    def static_requirements():
        return [WaveletProfile.static_name()]
    
    @staticmethod
    def static_parameters():
        return []
    
    def __init__(self, dataset: DataFrame, requirements: list[IFeatureGroup]):
        super().__init__()

        from ..features.decomposition import WaveletKP

        self.wave_kp: WaveletKP = WaveletKP()
        self.wmp_requirement: WaveletProfile = requirements[0]


    def name(self):
        return WaveletKProfile.static_name()
    
    def feature_names(self):
        return WaveletKProfile.static_feature_names()
    
    def requirements(self):
        return WaveletKProfile.static_requirements()
    
    def parameters(self):
        return WaveletKProfile.static_parameters()
    
    # def set_parameter(self, parameter_name: str, parameter_value: int | float | bool):
        # if parameter_name == TotalWaveletLevel.static_name():
        #     self.total_level.set(parameter_value)
        # elif parameter_name == AggregateWaveletLevel.static_name():
        #     self.aggregate_level.set(parameter_value)
        # elif parameter_name == WaveletType.static_name():
        #     self.wavelet_type.set(parameter_value)

        # elif parameter_name == UseCudaParam.static_name():
        #     self.use_cuda.set(parameter_value)
        # elif parameter_name == LeftOnlyParam.static_name():
        #     self.left_only.set(parameter_value)
        # elif parameter_name == SkipStartParam.static_name():
        #     self.skip_start.set(parameter_value)
        # elif parameter_name == ExclusionZoneRatioParam.static_name():
        #     self.exclusion_zone.set(parameter_value)
        # elif parameter_name == QuantileParam.static_name():
        #     self.quantile.set(parameter_value)


    def get_feature(self, name) -> np.ndarray:
        return self.wave_kp.feature
    
    def update(self, dataset, window_size: int):
        
        mps = self.wmp_requirement.wave_mp.feature
        self.wave_kp.feature = np.sort(mps, axis=0)[::-1]