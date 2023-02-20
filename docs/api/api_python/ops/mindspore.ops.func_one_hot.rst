mindspore.ops.one_hot
=====================

.. py:function:: mindspore.ops.one_hot(indices, depth, on_value, off_value, axis=-1)

    返回一个one-hot类型的Tensor。

    生成一个新的Tensor，由索引 `indices` 表示的位置取值为 `on_value` ，而在其他所有位置取值为 `off_value` 。

    .. note::
        如果输入索引为秩 `n` ，则输出为秩 `n+1` 。新轴在 `axis` 处创建。

    参数：
        - **indices** (Tensor) - 输入索引，shape为 :math:`(X_0, \ldots, X_n)` 的Tensor。数据类型必须为int32或int64。
        - **depth** (int) - 输入的Scalar，定义one-hot的深度。
        - **on_value** (Union[Tensor, int, float]) - 当 `indices[j] = i` 时，用来填充输出的值。数据类型为float16或float32。
        - **off_value** (Union[Tensor, int, float]) - 当 `indices[j] != i` 时，用来填充输出的值。数据类型与 `on_value` 的相同。
        - **axis** (int) - 指定one-hot的计算维度。例如，如果 `indices` 的shape为 :math:`(N, C)` ，`axis` 为-1，则输出shape为 :math:`(N, C, depth)` ，如果 `axis` 为0，则输出shape为 :math:`(depth, N, C)` 。默认值：-1。

    返回：
        Tensor，one-hot类型的Tensor。shape为 :math:`(X_0, \ldots, X_{axis}, \text{depth} ,X_{axis+1}, \ldots, X_n)` 。

    异常：
        - **TypeError** - `axis` 或 `depth` 不是int。
        - **TypeError** - `indices` 的数据类型既不是uint8，也不是int32或者int64。
        - **TypeError** - `indices`、`on_value` 或 `off_value` 不是Tensor。
        - **ValueError** - `axis` 不在[-1, ndim]范围内。ndim为 `indices` 的维度。
        - **ValueError** - `depth` 小于0。
    