mindspore.numpy.around
======================

.. py:function:: mindspore.numpy.around(a, decimals=0)

    向给定的小数位数四舍五入。

    .. note::
        不支持NumPy参数 `out` 。 不支持复数。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 输入数据。
        - **decimals** (int) - 需要四舍五入到的小数位数。 默认值： `0` 。

    返回：
        Tensor。 一个与 `a` 同类型的Tensor，包含四舍五入后的数值。 float类型的数值四舍五入后的结果仍是float类型。

    异常：
        - **TypeError** - 如果输入不能被转化为一个Tensor或参数 `decimals` 不是整数。