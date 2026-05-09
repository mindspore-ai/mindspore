mindspore.numpy.array_equal
=================================

.. py:function:: mindspore.numpy.array_equal(a1, a2, equal_nan=False)

    当输入数组shape相同且所有元素相等时，返回 `True`。

    .. note::
        在MindSpore中，会返回一个bool tensor，因为在图模式下，该值无法在编译时追踪和计算。由于Ascend平台对 ``nan`` 的处理不同，目前在Ascend上不支持 ``equal_nan`` 参数。

    参数：
        - **a1** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **a2** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **equal_nan** (bool，可选) - 是否将 ``NaN`` 视为相等。默认值： ``False`` 。

    返回：
        标量bool tensor，如果输入相等，值为 ``True`` ，否则为 ``False`` 。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。