mindspore.ops.polar
===================

.. py:function:: mindspore.ops.polar(abs, angle)

    将极坐标转化为笛卡尔坐标。

    返回一个复数tensor，其元素是由输入极坐标构造的笛卡尔坐标。其中极坐标由极径 `abs` 和极角 `angle` 给定。

    .. math::

        y_{i} = abs_{i} * \cos(angle_{i}) + abs_{i} * \sin(angle_{i}) * j

    参数：
        - **abs** (Tensor, float) - 极径。
        - **angle** (Tensor, float) - 极角。

    返回：
        Tensor
