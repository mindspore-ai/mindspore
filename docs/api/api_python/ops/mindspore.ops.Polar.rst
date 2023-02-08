mindspore.ops.Polar
====================

.. py:class:: mindspore.ops.Polar(abs, angle)

    将极坐标转化为笛卡尔坐标。

    返回一个复数Tensor，其元素是与输入极坐标对应的笛卡尔坐标。

    .. math::

        y_{i} =  abs_{i} * cos(angle_{i}) + abs_{i} * sin(angle_{i}) * j

    输入：
        - **abs** (Tensor) - 极径。Tensor的shape为 :math:`(N,*)` ，其中 :math:`N` 为输入Tensor的批量大小， :math:`*` 为任意数量的额外维度。其数据类型须为：float32、float64。
        - **angle** (Tensor) - 极角。其shape与数据类型与 `abs` 一致。

    输出：
        Tensor。其shape与数据类型与 `abs` 一致。

    异常：
        - **TypeError** - `abs` 或 `angle` 不是Tensor。
        - **TypeError** - 输入数据类型不是float32或float64。
        - **TypeError** - `abs` 和 `angle` 数据类型不一致。
        - **ValueError** - `abs` 和 `angle` 的shape不一致。
