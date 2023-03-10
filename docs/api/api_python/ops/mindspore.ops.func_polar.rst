mindspore.ops.polar
===================

.. py:function:: mindspore.ops.polar(abs, angle)

    将极坐标转化为笛卡尔坐标。

    返回一个复数Tensor，其元素是由输入极坐标构造的笛卡尔坐标。其中极坐标由极径 `abs` 和极角 `angle` 给定。

    .. math::

        y_{i} =  abs_{i} * cos(angle_{i}) + abs_{i} * sin(angle_{i}) * j

    参数：
        - **abs** (Tensor) - 极径。Tensor的shape为 :math:`(N,*)` ，其中 :math:`N` 为输入Tensor的批量大小， :math:`*` 为任意数量的额外维度。其数据类型须为：float32、float64。
        - **angle** (Tensor) - 极角。其shape与数据类型与 `abs` 一致。

    返回：
        Tensor，其shape与 `abs` 一致。
        - 如果输入数据类型是float32，则输出类型为complex64。
        - 如果输入数据类型是float64，则输出类型为complex128。

    异常：
        - **TypeError** - `abs` 或 `angle` 不是Tensor。
        - **TypeError** - 输入数据类型不是float32或float64。
        - **TypeError** - `abs` 和 `angle` 数据类型不一致。
        - **ValueError** - `abs` 和 `angle` 的shape不一致。
