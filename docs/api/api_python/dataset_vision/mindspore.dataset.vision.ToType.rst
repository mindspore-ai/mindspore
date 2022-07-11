mindspore.dataset.vision.ToType
===============================

.. py:class:: mindspore.dataset.vision.ToType(data_type)

    将输入转换为指定的MindSpore数据类型或NumPy数据类型。

    效果同 :class:`mindspore.dataset.transforms.TypeCast` 。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    参数：
        - **data_type** (Union[mindspore.dtype, numpy.dtype]) - 输出图像的数据类型，例如 :class:`numpy.float32` 。

    异常：
        - **TypeError** - 当 `data_type` 的类型不为 :class:`mindspore.dtype` 或 :class:`numpy.dtype` 。
