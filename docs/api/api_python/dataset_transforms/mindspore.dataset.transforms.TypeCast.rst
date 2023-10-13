mindspore.dataset.transforms.TypeCast
=====================================

.. py:class:: mindspore.dataset.transforms.TypeCast(data_type)

    将输入的Tensor转换为指定的数据类型。

    .. note::
        此操作默认通过 CPU 执行，也支持异构加速到 GPU 或 Ascend 上执行。

    参数：
        - **data_type** (Union[mindspore.dtype, numpy.dtype]) - 指定要转换的数据类型。

    异常：      
        - **TypeError** - 当 `data_type` 的类型不为 :class:`mindspore.dtype` 或 :class:`numpy.dtype` 。
