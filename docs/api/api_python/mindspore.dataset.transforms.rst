mindspore.dataset.transforms
============================

This module is to support common augmentations. C_transforms is a high performance image augmentation module which is developed with C++ OpenCV. Py_transforms provide more kinds of image augmentations which are developed with Python PIL.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms
    from mindspore.dataset.transforms import py_transforms

mindspore.dataset.transforms.c_transforms
-----------------------------------------

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.c_transforms.Compose
    mindspore.dataset.transforms.c_transforms.Concatenate
    mindspore.dataset.transforms.c_transforms.Duplicate
    mindspore.dataset.transforms.c_transforms.Fill
    mindspore.dataset.transforms.c_transforms.Mask
    mindspore.dataset.transforms.c_transforms.OneHot
    mindspore.dataset.transforms.c_transforms.PadEnd
    mindspore.dataset.transforms.c_transforms.RandomApply
    mindspore.dataset.transforms.c_transforms.RandomChoice
    mindspore.dataset.transforms.c_transforms.Relational
    mindspore.dataset.transforms.c_transforms.Slice
    mindspore.dataset.transforms.c_transforms.TypeCast
    mindspore.dataset.transforms.c_transforms.Unique

mindspore.dataset.transforms.py_transforms
------------------------------------------

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.py_transforms.Compose
    mindspore.dataset.transforms.py_transforms.OneHotOp
    mindspore.dataset.transforms.py_transforms.RandomApply
    mindspore.dataset.transforms.py_transforms.RandomChoice
    mindspore.dataset.transforms.py_transforms.RandomOrder
