.. _pyfade_structure:

PyFADe Structure
================

The **PyFADe** library is architected around three core architectural layers. Data flows sequentially from raw time-series into rolling windows, which then drive feature calculations.

---

Series Module
-------------

Responsible for ingesting and holding raw one-dimensional or multi-dimensional time-series data. 

* **Core Class:** :class:`DataFrame <pyfade.series.DataFrame>`
* **Creational Pattern:** Must be instantiated via a :class:`DataFrameBuilder <pyfade.series.DataFrameBuilder>`.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyfade

    # Ingest 1D signal data
    n = np.arange(50, dtype=float)
    y = np.sin(n * 0.5)

    # Initialize the DataFrame via its builder
    builder = pyfade.series.DataFrameBuilder(y)
    dataset = builder.build()

    # Visualize the base data structure
    fig, axs = dataset.plot()
    axs[0].grid(True)
    plt.show()


Window Module
-------------

Handles splitting your `DataFrame` into rolling, fixed-sized intervals where analytical computations take place. 

* **Core Class:** :class:`Window <pyfade.window.Window>`
* **Creational Pattern:** Configured and generated using a :class:`WindowBuilder <pyfade.window.WindowBuilder>`.

.. code-block:: python

    # Using the 'dataset' object created in the Series step
    window_builder = pyfade.window.WindowBuilder(dataset)
    window = window_builder.set_window_size(2).build()


.. _reference_group_module:

Group Module
------------

The execution layer for statistical computations. Features (like rolling means or standard deviations) are organized into structured **Feature Groups**. When a `DataFrame` updates, its parent `Window` triggers the attached `FeatureGroup` to recompute all child features.

* **Creational Pattern:** Attached to windows using the :class:`FeatureGroupBuilder <pyfade.group.FeatureGroupBuilder>`.

.. topic:: Complete Integration Example

The following example shows how the **Series**, **Window**, and **Group** modules link together to compute a rolling average:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyfade

    # 1. Pipeline Input (Series)
    y = np.sin(np.arange(50, dtype=float) * 0.5)
    dataset = pyfade.series.DataFrameBuilder(y).build()

    # 2. Pipeline Processing Window (Window)
    window = pyfade.window.WindowBuilder(dataset).set_window_size(2).build()

    # 3. Pipeline Calculation Engine (Group)
    group_builder = pyfade.group.FeatureGroupBuilder(
        pyfade.group.FeatureGroups.ContinuousStatistics, 
        window
    )
    group_builder.build()

    # Extract computed attributes
    print(window.mean)

    # Plot the specific engineered feature
    fig, axs = window.plot_feature(pyfade.features.Features.Mean)
    axs[0].grid(True)
    plt.show()