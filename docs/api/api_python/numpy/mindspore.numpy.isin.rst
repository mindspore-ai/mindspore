mindspore.numpy.isin
=================================

.. py:function:: mindspore.numpy.isin(element, test_elements, invert=False)

    在 `test_elements` 中的元素上计算，并仅在 `element` 上进行广播。返回一个与 `element` 形状相同的bool数组，其中 `element` 的元素在 `test_elements` 中时为True，否则为False。

    .. note::
        由于实现不依赖于输入数组的唯一性，因此不支持NumPy的 ``assume_unique`` 参数。

    参数：
        - **element** (Union[int, float, bool, list, tuple, Tensor]) - 输入数组。
        - **test_elements** (Union[int, float, bool, list, tuple, Tensor]) - 用于测试 `element` 中每个值的对比值。
        - **invert** (boolean, 可选) - 如果为True，返回数组中的值将取反，相当于计算 `element` 不在 `test_elements` 中的情况。默认值： ``False`` 。

    返回：
        Tensor，与 `element` 具有相同的shape。 `element[isin]` 的值在 `test_elements` 中。