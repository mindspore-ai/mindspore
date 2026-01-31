mindspore.numpy.where
=================================

.. py:function:: mindspore.numpy.where(condition, x=None, y=None)

    根据 `condition` 从 `x` 或 `y` 中选择元素。

    .. note::
        由于不支持 ``nonzero`` ， `x` 和 `y` 必须都是Tensor输入。

    参数：
        - **condition** (Tensor) - 当为 ``True`` 时，选择 `x` 中的值，否则选择 `y` 中的值。
        - **x** (Tensor，可选) - 选择值的来源。默认值： ``None`` 。
        - **y** (Tensor，可选) - 选择值的来源。 `x` ， `y` 和 `condition` 需要能够广播到相同的shape。默认值： ``None`` 。

    返回：
        Tensor或标量，其中 `condition` 为 ``True`` 的位置取自 `x` ，其他位置取自 `y` 。

    异常：
        - **ValueError** - 如果操作数不能广播到相同的shape。