.. _reference-own_features:

Creating Your Own Features
==========================

You can extend the **PyFADe** framework by implementing custom metrics. To do this, you must inherit from and implement two abstract base interfaces:

* :class:`pyfade.features.IFeature` — Defines an individual calculated metric.
* :class:`pyfade.group.IFeatureGroup` — Manages the lifecycle, requirements, parameters, and computation updates for a collection of related features.

Below is a complete guide to creating two custom features (``Meanpy`` and ``Stdpy``) and organizing them inside a custom Feature Group named ``PythonStatistics``.

---

1. Implementing Custom Features
-------------------------------

Each custom feature must provide a unique static name identity and instance mapping so the parent pipeline can register it correctly.

.. code-block:: python

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

---

2. Implementing the Feature Group
---------------------------------

The Feature Group coordinates initialization, checks for prerequisites, handles incoming parameters, and executes calculations inside the ``update`` method whenever the rolling dataset advances.

.. code-block:: python

    import numpy as np
    from pyfade.features import Features
    from pyfade.group import IFeatureGroup, ContinuousStatistics

    class PythonStatistics(IFeatureGroup):

        @staticmethod
        def static_name():
            return 'python_statistics'
        
        @staticmethod
        def static_feature_names():
            return [Features.Meanpy.value,
                    Features.Stdpy.value]
        
        @staticmethod
        def static_requirements():
            return [ContinuousStatistics.static_name()]
        
        @staticmethod
        def static_parameters():
            return []

        def __init__(self, dataset, requirements: list[IFeatureGroup]):
            super().__init__()

            from ..features.pystatistics import Meanpy, Stdpy

            self.meanpy: Meanpy = Meanpy()
            self.stdpy: Stdpy = Stdpy()

            self.cont_mean = requirements[0]

        def name(self):
            return PythonStatistics.static_name()
        
        def feature_names(self):
            return PythonStatistics.static_feature_names()
        
        def requirements(self):
            return PythonStatistics.static_requirements()
        
        def parameters(self):
            return PythonStatistics.static_parameters()

        def get_feature(self, name) -> np.ndarray:
            features = self.feature_names()
            if name == features[0]:
                return self.meanpy.feature
            elif name == features[1]:
                return self.stdpy.feature
            
        def update(self, dataset, window_size: int):
            self.meanpy.feature = self.cont_mean.means
            self.stdpy.feature = self.cont_mean.stds