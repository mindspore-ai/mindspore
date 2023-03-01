mindspore.Tensor.min
====================

.. py:method:: mindspore.Tensor.min(axis=None, keepdims=False, initial=None, where=True)

    返回Tensor元素中的最小值或沿 `axis` 轴方向上的最小值。

    参数：
        - **axis** (Union[None, int, list, tuple of ints], 可选) - 轴，在该轴方向上进行操作。默认情况下，使用扁平输入。如果该参数为整数元组，则在多个轴上选择最小值，而不是在单个轴或所有轴上进行选择。默认值：None。
        - **keepdims** (bool, 可选) - 如果这个参数为True，被删去的维度保留在结果中，且维度设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。默认值：False。
        - **initial** (scalar, 可选) - 输出元素的最小值。如果对空切片进行计算，则该参数必须设置。默认值：None。
        - **where** (Tensor[bool], 可选) - 一个bool类型的Tensor，被广播以匹配数组维度和选择包含在降维中的元素。如果传递了一个非默认值，则必须提供初始值。默认值：True。

    返回：
        Tensor或标量，输入Tensor的最小值。如果 `axis` 为None，则结果是一个标量值。如果提供了 `axis` ，则结果是Tensor ndim - 1维度的一个数组。

    异常：
        - **TypeError** - 参数的数据类型与上述不一致。
