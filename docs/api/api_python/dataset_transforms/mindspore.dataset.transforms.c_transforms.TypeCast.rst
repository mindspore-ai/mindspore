mindspore.dataset.transforms.c_transforms.TypeCast
==================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.TypeCast(data_type)

    将输入的Tensor转换为指定的数据类型。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    **参数：**

    - **data_type** (mindspore.dtype) - 指定要转换的数据类型。

    **异常：**
      
    - **TypeError** - 参数 `data_type` 类型不为MindSpore支持的数据类型 :class:`mindspore.dtype` 。
