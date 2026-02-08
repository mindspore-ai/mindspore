mindspore.mint.randn_like
=========================

.. py:function:: mindspore.mint.randn_like(input, *, dtype=None, device=None)

    返回shape与输入相同，类型为 `dtype` 的tensor。dtype由输入决定，其元素取值服从 :math:`[0, 1)` 区间内的正态分布。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的tensor。用来决定输出tensor的shape和默认的dtype。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`，可选) - 指定输出tensor的dtype，必须是float类型。如果是 ``None`` ，则使用输入tensor的dtype。默认值： ``None`` 。
        - **device** (str, 可选) - 指定tensor使用的内存来源。仅支持 ``"Ascend"`` 、 ``"npu"``。如果是 ``None`` ，则使用参数 `input` 的device。默认值 ``None`` 。

    返回：
        Tensor，shape和dtype由输入决定其元素为服从标准正态分布的数字。

    异常：
        - **RuntimeError** - 如果 `input` 的device是 ``"CPU"`` ，同时 `device` 是 ``None`` 。
        - **RuntimeError** - 如果 `device` 是 ``"CPU"`` 。
        - **ValueError** - 如果 `device` 是 ``"GPU"`` 。
