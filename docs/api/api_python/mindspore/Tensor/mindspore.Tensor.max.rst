mindspore.Tensor.max
====================

.. py:method:: mindspore.Tensor.max(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)

    返回Tensor的最大值或轴方向上的最大值。

    参数：
        - **axis** (Union[None, int, list, tuple of ints], 可选) - 轴，在该轴方向上进行操作。默认情况下，使用扁平输入。如果该参数为整数元组，则在多个轴上选择最大值，而不是在单个轴或所有轴上进行选择。默认值：None。
        - **keepdims** (bool, 可选) - 如果这个参数为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。默认值：False。

    关键字参数：
        - **initial** (scalar, 可选) - 输出元素的最小值。如果对空切片进行计算，则该参数必须设置。默认值：None。
        - **where** (bool Tensor, 可选) - 一个bool数组，被广播以匹配数组维度和选择包含在降维中的元素。如果传递了一个非默认值，则还必须提供初始值。默认值：True。
        - **return_indices** (bool, 可选) - 是否返回最大值的下标。默认值：False。如果 `axis` 是 一个list或一个int类型的tuple, 则必须取值为False。

    返回：
        Tensor或标量，输入Tensor的最大值。如果 `axis` 为None，则结果是一个标量值。如果提供了 `axis` ，则结果是Tensor ndim - 1维度的一个数组。

    异常：
        - **TypeError** - 参数具有前面未指定的类型。
