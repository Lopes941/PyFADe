
import numpy as np
import pandas
import pywt

import matplotlib.pyplot as plt

from pyfade.group import IFeatureGroup,  IFeatureGroupParameter
from pyfade.group import UseCudaParam, LeftOnlyParam, SkipStartParam, ExclusionZoneRatioParam, QuantileParam

from pyfade.features import Features
from pyfade.series import DataFrame

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

            sig_A = np.interp(index.astype('int64'),new_ind_A.values.astype('int64'),sig_A)
            sig_D = np.interp(index.astype('int64'),new_ind_D.values.astype('int64'),sig_D)

            #Writing decomposed signals
            self.decomp[lvl] = [pandas.Series(sig_A,index=index), pandas.Series(sig_D,index=index)]
            self.decomp[lvl][0].name = f'Approximation {lvl+1}'
            self.decomp[lvl][1].name = f'Detail {lvl+1}'

    def _build_from_coeffs(self, index, coeffs):

        n_levels = len(coeffs) - 1
        wavelet = self.wavelet
        self.decomp = [None] * n_levels

        coeffs_A = [coeffs[0]] + [None] * n_levels
        sig_A_full = pywt.waverec(coeffs_A, wavelet)
        sig_A_full = np.interp(index.astype('int64'),
                            np.linspace(index.min().value, index.max().value, len(sig_A_full)),
                            sig_A_full)
        
        for lvl in range(n_levels):

            # Zero all coefficients except D at the current level
            coeffs_D = [np.zeros_like(c) if c is not None else None for c in coeffs]
            coeffs_D[lvl + 1] = coeffs[lvl + 1] if coeffs[lvl + 1] is not None else np.zeros_like(coeffs[0])
            sig_D = pywt.waverec(coeffs_D, wavelet)

            # Interpolate reconstructed detail signal to match index
            sig_D = np.interp(index.astype('int64'),
                            np.linspace(index.min().value, index.max().value, len(sig_D)),
                            sig_D)

            # Approximation signal at this level (optional)
            # Build an approximation version for this level as well
            coeffs_A = [None] * (n_levels + 1)
            coeffs_A[0] = coeffs[0]
            for j in range(1, lvl + 1):
                coeffs_A[j] = coeffs[j]
            sig_A = pywt.waverec(coeffs_A, wavelet)
            sig_A = np.interp(index.astype('int64'),
                            np.linspace(index.min().value, index.max().value, len(sig_A)),
                            sig_A)

            # Store in a structure similar to your previous version
            self.decomp[n_levels - lvl-1] = [
                pandas.Series(sig_A, index=index, name=f"Approximation {n_levels - lvl}"),
                pandas.Series(sig_D, index=index, name=f"Detail {n_levels - lvl}")
            ]


    def __init__(self, data: np.ndarray, index, wavelet: str, level: int, coefficients: list = None):

        self.wavelet = wavelet
        self.level = level
        if coefficients is None:
            self._decompose_signal(data,index)
        else:
            self._build_from_coeffs(index,coefficients)

    def get_dataframe_from_decomp(self):

        data = [None]*(self.level+1)
        names = [None]*(self.level+1)
        for lvl in range(self.level):
            data[lvl] = self.decomp[lvl][1].values
            names[lvl] = f'Detail {lvl+1}'

        index = self.decomp[-1][0].index
        data[lvl+1] = self.decomp[lvl][0].values
        names[lvl+1] = f'Approximation {lvl+1}'

        frame = pandas.DataFrame(np.array(data).T, index=index, columns=names)

        return frame

    
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

class WaveletAnalysisType(IFeatureGroupParameter):

    @staticmethod
    def static_name():
        return 'wavelet_analysis_type'
    
    def __init__(self, val: int | float | bool = 'kp'):
        super().__init__()
        self.set(val)

    def name(self):
        return WaveletAnalysisType.static_name()

    def set(self, value):
        self.value = value

class WaveletDenoiseThreshold(IFeatureGroupParameter):

    @staticmethod
    def static_name():
        return 'wavelet_denoise_threshold'
    
    def __init__(self, val: int | float | bool = 0.05):
        super().__init__()
        self.set(val)

    def name(self):
        return WaveletDenoiseThreshold.static_name()

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
                AggregateWaveletLevel.static_name(),
                WaveletType.static_name(),
                WaveletAnalysisType.static_name(),
                WaveletDenoiseThreshold.static_name(),
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
        self.analysis_type: WaveletAnalysisType = WaveletAnalysisType()
        self.denoise_threshold: WaveletDenoiseThreshold = WaveletDenoiseThreshold()

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
        elif parameter_name == WaveletAnalysisType.static_name():
            self.analysis_type.set(parameter_value)
        elif parameter_name == WaveletDenoiseThreshold.static_name():
            self.denoise_threshold.set(parameter_value)

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
        from ..window import WindowBuilder
        from ..group import FeatureGroupBuilder, FeatureGroups
        
        self.decomposition: list = [None]*dataset.ndim
        mp_params = {
            'use_cuda': self.use_cuda.get(),
            'left_only': self.left_only.get(),
            'skip_start': self.skip_start.get(),
            'quantile_threshold': self.quantile.get(),
            'exclusion_zone_ratio': self.exclusion_zone.get()
        }

        if self.analysis_type.value == "last":

            # Decomposing signals
            for i in range(dataset.ndim):
                self.decomposition[i] = Decomposition(dataset.data[i,:],dataset.index,self.wavelet_type.value, self.total_level.value)

            # Getting MPs from each signal
            self.wave_mp.feature = np.zeros((dataset.ndim, dataset.len-window_size+1))
            for i in range(dataset.ndim):

                data = self.decomposition[i].decomp[-1][0].values

                frame = DataFrameBuilder(data).build()
                window = WindowBuilder(frame).set_window_size(window_size).build()
                from ..group import FeatureGroups
                FeatureGroupBuilder(FeatureGroups.MatrixProfile,window).set_parameters(mp_params).build()

                self.wave_mp.feature[i,:] = window.mp

        elif self.analysis_type.value == "kp":

            # Decomposing signals
            for i in range(dataset.ndim):
                self.decomposition[i] = Decomposition(dataset.data[i,:],dataset.index,self.wavelet_type.value, self.total_level.value)

            # Getting MPs from each signal
            self.wave_mp.feature = np.zeros((dataset.ndim, dataset.len-window_size+1))
            for i in range(dataset.ndim):

                # Creating dataframe for each value
                data = np.zeros((self.total_level.value, dataset.len))
                for lvl in range(self.total_level.value):
                    data[lvl,:] = self.decomposition[i].decomp[lvl][1].values
                # data[-1,:] = self.decomposition[i].decomp[-1][0].values

                frame = DataFrameBuilder(data).build()
                window = WindowBuilder(frame).set_window_size(window_size).build()
                from ..group import FeatureGroups
                FeatureGroupBuilder(FeatureGroups.MatrixProfile,window).set_parameters(mp_params).build()
                FeatureGroupBuilder(FeatureGroups.KProfile,window).build()

                self.wave_mp.feature[i,:] = window.kp[self.aggregate_level.value,:]

        elif self.analysis_type.value == "diff":

            # Decomposing signals
            for i in range(dataset.ndim):
                self.decomposition[i] = Decomposition(dataset.data[i,:],dataset.index,self.wavelet_type.value, self.total_level.value)



            # Getting data difference from each signal
            difference = -dataset.data.copy()
            for i in range(dataset.ndim):

                # Getting decomposition difference
                difference[i,:] += self.decomposition[i].get_dataframe_from_decomp().sum(axis=1)

                plt.figure()
                plt.plot(difference[i,:])


            frame = DataFrameBuilder(difference).build()
            window = WindowBuilder(frame).set_window_size(window_size).build()
            from ..group import FeatureGroups
            FeatureGroupBuilder(FeatureGroups.MatrixProfile,window).set_parameters(mp_params).build()

            self.wave_mp.feature = window.mp

        elif self.analysis_type.value == "denoise":

            # Denoising each signal
            denoised = -dataset.data*0
            coeffs = [None]*dataset.ndim
            for i in range(dataset.ndim):

                coeffs[i] = pywt.wavedec(dataset.data[i,:], self.wavelet_type.value, level=self.total_level.value)

                # Getting decomposition denoise
                coeffs_thresh = [pywt.threshold(c, value=self.denoise_threshold.value, mode='soft') for c in coeffs[i]]


                reconstructed = pywt.waverec(coeffs_thresh, self.wavelet_type.value)


                # Match length to original
                reconstructed = reconstructed[:dataset.data.shape[1]]  # Trim if longer
                # or pad if shorter
                if len(reconstructed) < dataset.data.shape[1]:
                    reconstructed = np.pad(reconstructed, (0, dataset.data.shape[1] - len(reconstructed)))
                    
                denoised[i, :] = reconstructed

                self.decomposition[i] = Decomposition(dataset.data[i,:], dataset.index, self.wavelet_type.value, self.total_level.value, coeffs_thresh)


            frame = DataFrameBuilder(denoised).build()
            window = WindowBuilder(frame).set_window_size(window_size).build()
            from ..group import FeatureGroups
            FeatureGroupBuilder(FeatureGroups.MatrixProfile,window).set_parameters(mp_params).build()

            self.wave_mp.feature = window.mp

        else:
            raise RuntimeError("Analysis type must be either 'last' or 'kp'")


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
    

    def get_feature(self, name) -> np.ndarray:
        return self.wave_kp.feature
    
    def update(self, dataset, window_size: int):
        
        mps = self.wmp_requirement.wave_mp.feature
        self.wave_kp.feature = np.sort(mps, axis=0)[::-1]