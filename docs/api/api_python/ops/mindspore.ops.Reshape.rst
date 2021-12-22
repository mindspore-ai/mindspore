mindspore.ops.Reshape
======================

.. py:class:: mindspore.ops.Reshape()

    基于给定的shape，对输入Tensor进行重新排列。

    `input_shape` 最多只能有一个-1，在这种情况下，它可以从剩余的维度和输入的元素个数中推断出来。

    **输入：**

    - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
    - **input_shape** (tuple[int]) - 输入tuple由多个整数构成，如 :math:`(y_1, y_2, ..., y_S)` 。只支持常量值。

    **输出：**

    Tensor，其shape为 :math:`(y_1, y_2, ..., y_S)` 。

    **异常：**

    - **ValueError** - 给定的 `input_shape`，如果它有多个-1，或者除-1（若存在）之外的元素的乘积小于或等于0，或者无法被输入Tensor的shape的乘积整除，或者与输入的数组大小不匹配。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    >>> reshape = ops.Reshape()
    >>> output = reshape(input_x, (3, 2))
    >>> print(output)
    [[-0.1  0.3]
     [ 3.6  0.4]
     [ 0.5 -3.2]]
