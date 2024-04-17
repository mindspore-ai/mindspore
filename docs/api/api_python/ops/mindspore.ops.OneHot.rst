mindspore.ops.OneHot
====================

.. py:class:: mindspore.ops.OneHot(axis=-1)

    返回一个one-hot类型的Tensor。

    生成一个新的Tensor，由索引 `indices` 表示的位置取值为 `on_value` ，而在其他所有位置取值为 `off_value` 。

    .. note::
        如果输入索引为秩 `N` ，则输出为秩 `N+1` 。新轴在 `axis` 处创建。当执行设备是 Ascend 时，如果 `on_value` 为int64类型，则 `indices` 也必须为int64类型，且 `on_value` 和 `off_value` 的取值只能是1和0。

    参数：
     - **axis** (int) - 指定one-hot的计算维度。例如，如果 `indices` 的shape为 :math:`(N, C)` ，`axis` 为-1，则输出shape为 :math:`(N, C, D)` ，如果 `axis` 为0，则输出shape为 :math:`(D, N, C)` 。默认值： ``-1`` 。

    输入：
        - **indices** (Tensor) - 输入索引，shape为 :math:`(X_0, \ldots, X_n)` 的Tensor。数据类型必须为int32或int64。
        - **depth** (Union[int, Tensor]) - 输入的Scalar，定义one-hot的深度。
        - **on_value** (Tensor) - 当 `indices[j] = i` 时，用来填充输出的值。数据类型必须为int32、int64、float16或float32。
        - **off_value** (Tensor) - 当 `indices[j] != i` 时，用来填充输出的值。数据类型与 `on_value` 的相同。

    输出：
        Tensor，one-hot类型的Tensor。shape为 :math:`(X_0, \ldots, X_{axis}, \text{depth} ,X_{axis+1}, \ldots, X_n)` ，输出数据类型与 `on_value` 的相同。

    异常：
        - **TypeError** - `axis` 或 `depth` 不是int。
        - **TypeError** - `indices` 的数据类型不是int32或者int64。
        - **TypeError** - `on_value` 的数据类型不是int32、int64、float16或者float32。
        - **TypeError** - `indices`、`on_value` 或 `off_value` 不是Tensor。
        - **ValueError** - `axis` 不在[-1, ndim]范围内。
        - **ValueError** - `depth` 小于0。
    