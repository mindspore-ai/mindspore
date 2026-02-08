mindspore.Tensor.cumsum
=======================

.. py:method:: mindspore.Tensor.cumsum(dim, *, dtype=None) -> Tensor

    计算输入Tensor `self` 沿轴 `dim` 的累积和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    参数：
        - **dim** (int) - 累积和计算的轴。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出数据类型。如果不为 ``None``，则输入会转化为 `dtype`。这有利于防止数值溢出。如果为 ``None``，则输出和输入的数据类型一致。默认值： ``None`` 。

    返回：
        Tensor，和输入Tensor的shape相同。

    异常：
        - **ValueError** - 如果 `dim` 超出范围。

    .. py:method:: mindspore.Tensor.cumsum(axis=None, dtype=None) -> Tensor
        :noindex:

    计算输入Tensor `self` 沿轴 `axis` 的累积和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    .. note::
        目前Ascend平台上，对于静态shape的场景， `self` 的数据类型暂仅支持：int8、uint8、int32、float32和float16；对于动态shape的场景， `self` 的数据类型暂仅支持：int32、float32和float16。

    参数：
        - **axis** (int) - 累积和计算的轴。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出数据类型。如果不为 ``None``，则输入会转化为 `dtype`。这有利于防止数值溢出。如果为 ``None``，则输出和输入的数据类型一致。默认值： ``None`` 。

    返回：
        Tensor，和输入Tensor的shape相同。

    异常：
        - **ValueError** - 如果 `axis` 超出范围。
