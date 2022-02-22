mindspore.dataset.transforms
============================

此模块用于通用数据增强，包括 `c_transforms` 和 `py_transforms` 两个子模块。

`c_transforms` 是一个高性能数据增强模块，基于C++实现。

而 `py_transforms` 提供了一种基于Python和NumPy的实现方式。

在API示例中，常用的模块导入方法如下：

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
