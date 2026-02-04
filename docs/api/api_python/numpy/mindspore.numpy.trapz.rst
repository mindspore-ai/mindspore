mindspore.numpy.trapz
=====================

.. py:function:: mindspore.numpy.trapz(y, x=None, dx=1.0, axis=-1)

    使用复合梯形规则沿给定轴进行积分。

    沿给定轴对 y (x) 进行积分。

    参数：
        - **y** (Tensor) - 输入要积分的数组。
        - **x** (Union[int, float, bool, list, tuple, Tensor], 可选) - 与 `y` 值对应的样本点。如果 `x` 为 ``None`` ，则设样本点之间的间隔为 `dx` 。默认值： ``None`` 。
        - **dx** (scalar, 可选) - 当 `x` 为 ``None`` 时，样本点之间的间隔。默认值： ``1.0`` 。
        - **axis** (int, 可选) - 积分所沿的轴。 默认值： ``-1`` 。

    返回：
        元素为float的Tensor，使用梯形规则近似的定积分。

    异常：
        - **ValueError** - 如果 `axis` 超出范围 ``[-y.ndim, y.ndim)`` 。