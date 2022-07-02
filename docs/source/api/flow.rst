Flow
====

General flow object
-------------------
.. autoclass:: pocomc.flow.Flow
    :members:

RealNVP
-------

.. autoclass:: pocomc.maf.RealNVP
    :members:

Masked Autoregressive Flow
--------------------------

.. autoclass:: pocomc.maf.MAF
    :members:

Masked Autoregressive Density Estimator
---------------------------------------

.. autoclass:: pocomc.maf.MADE
    :members:


Masked Linear layer
-------------------

.. autoclass:: pocomc.maf.MaskedLinear
    :members:

Batch Normalisation layer
-------------------------

.. autoclass:: pocomc.maf.BatchNorm
    :members:

Linear Masked Coupling layer
----------------------------

.. autoclass:: pocomc.maf.LinearMaskedCoupling
    :members:


Flow Sequential
---------------

.. autoclass:: pocomc.maf.FlowSequential
    :members:


Helper function to create masks
-------------------------------

.. autofunction:: pocomc.maf.create_masks

Fitting function
----------------

.. autofunction:: pocomc.flow.fit