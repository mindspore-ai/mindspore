mindspore.ops.tensor_dot
=========================

.. py:function:: mindspore.ops.tensor_dot(x1, x2, axes)

    在指定轴上对Tensor `a` 和 `b` 进行点乘操作。

    点乘操作可以在指定轴上计算 `a` 和 `b` 元素的乘积和。轴数量必须由x1和x2指定，并且该值必须在 `a` 和 `b` 的维度数量的范围内。

    两个输入中选定的维度必须相互匹配。

    axes = 0为外积。axes = 1为普通矩阵乘法（输入是二维的）。axes = 1与axes = ((1,),(0,))相同，其中 `a` 和 `b` 都是二维的。axes = 2与axes = ((1,2),(0,1))相同，其中 `a` 和 `b` 都是三维的。

    参数：
        - **x1** (Tensor) - tensor_dot的第一个输入Tensor，其数据类型为float16或float32。
        - **x2** (Tensor) - tensor_dot的第二个输入Tensor，其数据类型为float16或float32。
        - **axes** (Union[int, tuple(int), tuple(tuple(int)), list(list(int))]) - 指定 `a` 和 `b` 计算轴，可为单个值，也可为长度为2的tuple或list。如果传递了单个值 `N` ，则自动从输入 `a` 的shape中获取最后N个维度，从输入 `b` 的shape中获取前N个维度，分别作为每个维度的轴。

    返回：
        Tensor，输出Tensor的shape为 :math:`(N + M)` 。其中 :math:`N` 和 :math:`M` 在两个输入中没有计算，是自由轴。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
        - **TypeError** - `axes` 不是int、tuple或list。