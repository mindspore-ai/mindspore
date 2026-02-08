mindspore.Tensor.max
====================

.. py:method:: mindspore.Tensor.max() -> Tensor

    返回输入Tensor的最大值。

    返回：
        Tensor，值为输入Tensor的最大值，类型与 `self` 相同。

    .. py:method:: mindspore.Tensor.max(dim, keepdim=False)
        :noindex:

    在给定轴上计算输入Tensor的最大值，并返回最大值和索引值。

    参数：
        - **dim** (int) - 指定计算维度。
        - **keepdim** (bool, 可选) - 表示是否保持维度，如果为 ``True`` ，输出将与输入保持相同的维度；如果为 ``False`` ，输出将减少维度。默认值： ``False`` 。

    返回：
        tuple(Tensor)，返回两个元素类型为Tensor的tuple，包含输入Tensor沿指定维度 `dim` 的最大值和相应的索引。

        - **values** (Tensor) - 输入Tensor沿给定维度的最大值，数据类型和 `self` 相同，shape和 `index` 相同。
        - **index** (Tensor) - 输入Tensor的沿给定维度的最大值索引，数据类型为 `int64` 。如果 `keepdim` 为 ``True`` ，输出Tensor的shape是 :math:`(self_1, self_2, ..., self_{dim-1}, 1, self_{dim+1}, ..., self_N)` 。否则输出shape为 :math:`(self_1, self_2, ..., self_{dim-1}, self_{dim+1}, ..., self_N)` 。

    异常：
        - **TypeError** - 如果 `keepdim` 不是bool类型。
        - **TypeError** - 如果 `dim` 不是int类型。
        - **TypeError** - 如果输入Tensor的数据类型为Complex。

    .. py:method:: mindspore.Tensor.max(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
        :noindex:

    返回Tensor的最大值或轴方向上的最大值。

    .. note::
        `axis` 为 ``None`` 时， `keepdims` 及以后参数均不会生效，同时索引固定返回0。

    参数：
        - **axis** (Union[None, int, list, tuple of ints], 可选) - 轴，在该轴方向上进行操作。默认情况下，使用扁平输入。如果该参数为整数元组，则在多个轴上选择最大值，而不是在单个轴或所有轴上进行选择。默认值： ``None`` 。
        - **keepdims** (bool, 可选) - 如果这个参数为 ``True`` ，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。默认值： ``False`` 。

    关键字参数：
        - **initial** (scalar, 可选) - 输出元素的最小值。如果对空切片进行计算，则该参数必须设置。默认值： ``None`` 。
        - **where** (bool Tensor, 可选) - 一个bool数组，被广播以匹配数组维度和选择包含在降维中的元素。如果传递了一个非默认值，则还必须提供初始值。默认值： ``True`` 。
        - **return_indices** (bool, 可选) - 是否返回最大值的下标。默认值： ``False`` 。如果 `axis` 是list或int类型的tuple，则必须取值为 ``False`` 。

    返回：
        Tensor或标量，输入Tensor的最大值。如果 `axis` 为 ``None`` ，则结果是一个标量值。如果提供了 `axis` ，则结果是Tensor ndim - 1维度的一个数组。

    异常：
        - **TypeError** - 参数具有前面未指定的类型。

    .. seealso::
        - :func:`mindspore.Tensor.argmin` ：返回沿轴最小值的索引。
        - :func:`mindspore.Tensor.argmax` ：返回沿轴最大值的索引。
        - :func:`mindspore.Tensor.min` ：返回整个Tensor最小值或沿轴最小值。
