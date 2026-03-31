mindspore.ops.tensor_dot
=========================

.. py:function:: mindspore.ops.tensor_dot(x1, x2, axes)

    沿指定轴计算两个张量的点乘。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `axes` 将重命名为 `dims`。

    参数：
        - **x1** (Tensor) - 输入tensor。
        - **x2** (Tensor) - 输入tensor。
        - **axes** (Union[int, tuple(int), tuple(tuple(int)), list(list(int))]) - 指定轴。如果是整数 k ，则对 x1 的后 k 个轴和 x2 的前 k 个轴求和。如果提供的是列表或元组，则axes[0] 指定 x1 的轴，axes[1] 指定 x2 的轴。

    返回：
        Tensor
