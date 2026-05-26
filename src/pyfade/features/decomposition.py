from ..core import IFeature
from ..series.series import DataFrame

class WaveletMP(IFeature):

    @staticmethod
    def static_name():
        return 'wavelet_mp'
    
    def __init__(self):
        super().__init__()

    def name(self):
        return WaveletMP.static_name()
    
class WaveletKP(IFeature):

    @staticmethod
    def static_name():
        return 'wavelet_kp'
    
    def __init__(self):
        super().__init__()

    def name(self):
        return WaveletKP.static_name()