mindspore.numpy.in1d
=================================

.. py:function:: mindspore.numpy.in1d(ar1, ar2, invert=False)

    测试一维数组的每个元素是否也存在于第二个数组中。
    返回与 `ar1` 长度相同的boolean数组，当 `ar1` 的某个元素存在于 `ar2` 中时，该位置为True，否则为False。

    .. note::
        不支持Numpy的 `assume_unique` 参数，因为该实现不依赖于输入数组的唯一性。

    参数：
        - **ar1** (Union[int, float, bool, list, tuple, Tensor]) - shape为 ``(M,)`` 的输入数组。
        - **ar2** (Union[int, float, bool, list, tuple, Tensor]) - 用于测试 `ar1` 中每个值的数组。
        - **invert** (boolean, 可选) - 如果为 ``True``，返回数组中的值将被反转（即当 `ar1` 的元素在 `ar2` 中时为 ``False``，否则为 ``True``）。默认值： ``False`` 。

    返回：
        Tensor，shape为 ``(M,)`` 。值为 `ar1` 中元素是否存在于 `ar2` 中的真值。
