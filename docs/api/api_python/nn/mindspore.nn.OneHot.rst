mindspore.nn.OneHot
====================

.. py:class:: mindspore.nn.OneHot(axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mindspore.float32)

    返回一个one-hot类型的Tensor。

    参数 `indices` 表示的位置取值为on_value，其他所有位置取值为off_value。

    .. note::
        如果indices是n阶Tensor，那么返回的one-hot Tensor则为n+1阶Tensor。

    如果 `indices` 是Scalar，则输出shape将是长度为 `depth` 的向量。

    如果 `indices` 是长度为 `features` 的向量，则输出shape为：

    .. code-block::

        features * depth if axis == -1

        depth * features if axis == 0

    如果 `indices` 是shape为 `[batch, features]` 的矩阵，则输出shape为：

    .. code-block::

        batch * features * depth if axis == -1

        batch * depth * features if axis == 1

        depth * batch * features if axis == 0

    **参数：**

    - **axis** (int) - 指定第几阶为depth维one-hot向量，如果轴为-1，则 features x depth，如果轴为0，则 depth x features。默认值：-1。
    - **depth** (int) - 定义one-hot向量的维度深度。默认值：1。
    - **on_value** (float) - one-hot值，当indices[j] = i时，填充output[i][j]的取值。默认值：1.0。
    - **off_value** (float) - 非one-hot值，当indices[j] != i时，填充output[i][j]的取值。默认值：0.0。
    - **dtype** (:class:`mindspore.dtype`) - 是'on_value'和'off_value'的数据类型，而不是索引的数据类型。默认值：mindspore.float32。

    **输入：**

    **indices** (Tensor) - 输入索引，任意维度的Tensor，数据类型为int32或int64。

    **输出：**

    Tensor，数据类型 `dtype` 的独热Tensor，维度为 `axis` 扩展到 `depth`，并填充on_value和off_value。`Outputs` 的维度等于 `indices` 的维度加1。

    **异常：**

    - **TypeError** - `axis` 或 `depth` 不是整数。
    - **TypeError** - `indices` 的dtype既不是int32，也不是int64。
    - **ValueError** - 如果 `axis` 不在范围[-1, len(indices_shape)]内。
    - **ValueError** - `depth` 小于0。