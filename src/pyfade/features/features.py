
from enum import Enum
from .pystatistics import Meanpy, Stdpy
from .decomposition import WaveletMP, WaveletKP


from ..core import IFeature, ContinuousStatistics, MatrixProfile, KProfile

class Features(Enum):
    Mean = ContinuousStatistics.static_feature_names()[0]
    StdDev = ContinuousStatistics.static_feature_names()[1]
    MatProfileVal = MatrixProfile.static_feature_names()[0]
    MatProfileInd = MatrixProfile.static_feature_names()[1]
    KProfileVal = KProfile.static_feature_names()[0]
    KProfileInd = KProfile.static_feature_names()[1]
    Meanpy = Meanpy.static_name()
    Stdpy = Stdpy.static_name()
    WaveletMP = WaveletMP.static_name()
    WaveletKP = WaveletKP.static_name()
