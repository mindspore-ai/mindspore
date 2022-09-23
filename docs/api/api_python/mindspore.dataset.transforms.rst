mindspore.dataset.transforms
============================

此模块用于通用数据增强，其中一部分增强操作是用C++实现的，具有较好的高性能，另一部分是基于Python实现，使用了NumPy模块作为支持。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

注意：旧的API导入方式已经过时且会逐步废弃，因此推荐使用上面的方式，但目前仍可按以下方式导入：

.. code-block::

    from mindspore.dataset.transforms import c_transforms
    from mindspore.dataset.transforms import py_transforms

更多详情请参考 `通用数据处理与增强 <https://www.mindspore.cn/tutorials/zh-CN/r1.9/advanced/dataset/augment_common_data.html>`_ 。

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。

变换
-----

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.Compose
    mindspore.dataset.transforms.Concatenate
    mindspore.dataset.transforms.Duplicate
    mindspore.dataset.transforms.Fill
    mindspore.dataset.transforms.Mask
    mindspore.dataset.transforms.OneHot
    mindspore.dataset.transforms.PadEnd
    mindspore.dataset.transforms.RandomApply
    mindspore.dataset.transforms.RandomChoice
    mindspore.dataset.transforms.RandomOrder
    mindspore.dataset.transforms.Slice
    mindspore.dataset.transforms.TypeCast
    mindspore.dataset.transforms.Unique

工具
-----

.. mscnautosummary::
    :toctree: dataset_transforms
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.transforms.Relational
