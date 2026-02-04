mindspore.Tensor.delete\_
=========================

.. py:method:: mindspore.Tensor.delete_(non_blocking=False)

    主动释放tensor在 `device` 或 `host` 侧的内存。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **non_blocking** (bool, 可选) - 内存异步释放。如果是 ``True`` ，内存异步释放。如果是 ``False`` ，内存同步释放。默认值：``False``。


    返回：
        Tensor，返回被修改后的 `self` 自身，其数据内存已被释放。
