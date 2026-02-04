mindspore.numpy.deg2rad
=======================

.. py:function:: mindspore.numpy.deg2rad(x, dtype=None)

    将角度从角度制转换为弧度制。

    参数：
        - **x** (Tensor) - 角度，单位为度。
        - **dtype** (mindspore.dtype, 可选) - 默认值： ``None`` 。覆盖输出Tensor的dtype。

    返回：
        Tensor，弧度制下的对应角度。 如果 `x` 是Tensor标量，则结果也是Tensor标量。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。