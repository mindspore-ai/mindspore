mindspore.dataset.transforms.TypeCast
=====================================

.. py:class:: mindspore.dataset.transforms.TypeCast(data_type)

    将输入的Tensor转换为指定的数据类型。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    参数：
        - **data_type** (Union[mindspore.dtype, numpy.dtype]) - 指定要转换的数据类型。

    异常：      
        - **TypeError** - 当 `data_type` 的类型不为 :class:`mindspore.dtype` 或 :class:`numpy.dtype` 。
