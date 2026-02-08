mindspore.mint.randn
====================

.. py:function:: mindspore.mint.randn(*size, generator=None, dtype=None, device=None)

    返回一个tensor，shape和dtype由输入决定，其元素为服从标准正态分布的数字。

    参数：
        - **size** (Union[int, tuple(int), list(int)]) - 输出的tensor的shape，例如，:math:`(2, 3)` or :math:`2`。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。
        - **dtype** (:class:`mindspore.dtype`，可选) - 输出tensor的dtype。如果是None， `mindspore.float32` 会被使用。默认值： ``None`` 。
        - **device** (str, 可选) - 指定tensor使用的内存来源。仅支持 ``"Ascend"`` 、 ``"npu"`` 。
          如果是 ``None`` ，则使用 :func:`mindspore.set_device` 设置的值。默认值 ``None`` 。

    返回：
        Tensor，shape和dtype由输入决定其元素为服从标准正态分布的数字。

    异常：
        - **ValueError** - 如果 `size` 包含负数。
        - **ValueError** - 如果 `device` 是 ``"GPU"`` 。
        - **RuntimeError** - 如果 `device` 是 ``"CPU"`` 。
