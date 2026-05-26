
from pyfade.features import IFeature

class Meanpy(IFeature):

    @staticmethod
    def static_name():
        return 'meanpy'
    
    def __init__(self):
        super().__init__()
    
    def name(self):
        return Meanpy.static_name()
    

class Stdpy(IFeature):

    @staticmethod
    def static_name():
        return 'stdpy'
    
    def __init__(self):
        super().__init__()
    
    def name(self):
        return Stdpy.static_name()

