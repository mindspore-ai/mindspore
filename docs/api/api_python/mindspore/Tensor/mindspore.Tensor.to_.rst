mindspore.Tensor.to\_
=====================

.. py:method:: mindspore.Tensor.to_(device=None, non_blocking=False)

    :func:`mindspore.Tensor.to` 的in-place版本，将原始tensor的设备转换成指定的 `device` 并返回。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **device** (str, 可选) - 用于指定输出tensor所在的硬件设备。默认值是 ``None`` 。
        - **non_blocking** (bool, 可选) - 数据异步转换。如果是 ``True`` ，数据类型异步转换。如果是 ``False`` ，数据同步转换。默认值：``False``。

    返回：
        Tensor，返回被修改后的 `self` 自身，其所在的设备为 `device`。
