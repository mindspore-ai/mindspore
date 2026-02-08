mindspore.mint.rand_like
========================

.. py:function:: mindspore.mint.rand_like(input, *, dtype=None, device=None)

    返回shape与输入相同，类型为 `dtype` 的Tensor，dtype由输入决定，其元素取值服从 :math:`[0, 1)` 区间内的均匀分布。

    参数：
        - **input** (Tensor) - 输入的Tensor，用来决定输出Tensor的shape和默认的dtype。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定的输出Tensor的dtype，必须是float类型。如果是 ``None`` ，`input` 的dtype会被使用。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。仅支持 ``"Ascend"`` 、 ``"npu"``。如果为 ``None`` ，将会使用 `input` 的device。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype由输入决定其元素为服从均匀分布的 :math:`[0, 1)` 区间的数字。
    
    异常：
        - **RuntimeError** - 如果 `input` 的device是 ``"CPU"`` ，同时 `device` 是 ``None`` 。
        - **RuntimeError** - 如果 `device` 是 ``"CPU"`` 。
        - **ValueError** - 如果 `device` 是 ``"GPU"`` 。
