mindspore.Tensor.is_shared
==========================

.. py:method:: mindspore.Tensor.is_shared()

    判断当前tensor是否位于共享内存。

    .. note::
        对于Ascend tensor，固定返回 ``True``。

    返回：
        Bool。如果该tensor位于共享内存，则返回 ``True``；否则返回 ``False``。