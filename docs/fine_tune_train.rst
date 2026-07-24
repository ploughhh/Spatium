Spatium Fine-tuning Training Example
========================

spaProFormer supports fine-tuning on multiple downstream tasks for spatial
proteomics analysis. Users can configure the downstream objective through the
``task`` parameter.


Selecting downstream tasks
--------------------------

The fine-tuning task is specified in the configuration dictionary:

.. code-block:: python

   "task": "cell_type_prediction"


Supported tasks include:

- ``cell_type_prediction``

  Predict cell types from spatial proteomics profiles.

- ``Prototype_classification``

  Perform prototype-based classification.

- ``neighborhood_identify``

  Predict spatial neighborhood composition.

- ``panel_expansion_continuous_new``

  Perform continuous protein panel expansion and imputation.

- ``image_integration``

  Integrate spatial proteomics data with image features.

- ``reconstruction``

  Reconstruct masked protein expression profiles.

- ``label_transfer``

  Transfer annotations from reference datasets.


Dataset input
-------------

The input data should be stored as Zarr format and provided through
``zarr_path``.

A single dataset can be provided:

.. code-block:: python

   zarr_path = "path/to/dataset.zarr"


For predefined training and validation datasets, provide multiple Zarr paths:

.. code-block:: python

   zarr_path = [
       "path/to/train.zarr",
       "path/to/validation.zarr"
   ]


When multiple files are provided, enable file-based splitting:

.. code-block:: python

   split_by_file = True


For a single dataset with internal splitting:

.. code-block:: python

   split_by_file = False


Complete fine-tuning example
----------------------------

The complete training script is shown below.

This example demonstrates:

- loading a pretrained spaProFormer model
- configuring downstream tasks
- preparing Zarr datasets
- initializing the fine-tuning model
- training with PyTorch Lightning


.. literalinclude:: examples/fine_tune/fine_tune_train.py
   :language: python
   :linenos: