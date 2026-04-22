mindspore.Tensor.scatter_add
============================

.. py:method:: mindspore.Tensor.scatter_add(dim, index, src) -> Tensor

    将 `src` 中所有的元素添加到 `self` 中 `index` 指定的索引处。
    其中 `dim` 控制scatter_add操作的轴。
    `self` 、 `index` 、 `src` 三者的rank都必须大于或等于1。

    下面看一个三维的例子：

    .. code-block::

        self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0

        self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1

        self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：删除 `scatter_add(indices, updates)` 重载，仅支持 `scatter_add(dim, index, src)`。

    .. note::
        输入Tensor `self` 的rank必须大于等于1。

    参数：
        - **dim** (int) - `self` 执行scatter_add操作的轴。取值范围是[-r, r)，其中r是 `self` 的rank。
        - **index** (Tensor) - `self` 要进行scatter_add操作的目标索引。数据类型为int32或int64，rank必须和 `self` 一致。除了 `dim` 指定的维度， `index` 的每一维的size都需要小于等于 `self` 对应维度的size。
        - **src** (Tensor) - 指定与 `self` 进行scatter_add操作的Tensor，其数据类型与 `self` 类型相同，shape中每一维的size必须大于等于 `index` 。

    返回：
        Tensor，shape和数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `index` 的数据类型不满足int32或int64。
        - **ValueError** - `self` 、 `index` 和 `src` 中，任意一者的rank小于1。
        - **ValueError** - `self` 、 `index` 和 `src` 的rank不一致。
        - **ValueError** - 除了 `dim` 指定的维度， `index` 的任意维的size大于 `self` 对应维度的size。
        - **ValueError** - `src` 任意维度size小于 `index` 对应维度的size。

    .. py:method:: mindspore.Tensor.scatter_add(indices, updates) -> Tensor
        :noindex:

    根据指定的更新值 `updates` 和输入索引 `indices` ，通过相加运算更新输入Tensor的值。当同一索引有不同值时，更新的结果将是所有值的总和。此操作与 :func:`mindspore.ops.scatter_nd_add` 类似，但更新后的结果是返回一个新的输出Tensor，而不是直接更新 `self` 。

    `indices` 的最后一个轴是每个索引向量的深度。对于每个索引向量， `updates` 中必须有相应的值。 `updates` 的shape应该等于 `self[indices]` 的shape。有关更多详细信息，请参见样例。

    .. math::
        output\left [indices  \right ] = input\_x + update

    .. note::
        输入Tensor `self` 的维度必须不小于 `indices.shape[-1]` 。

        如果 `indices` 中的值超出输入 `self` 索引范围：

        - GPU平台上相应的 `updates` 不会更新到 `self` 且不会抛出索引错误。
        - CPU平台上直接抛出索引错误。
        - Ascend平台不支持越界检查，若越界可能会造成未知错误。

    参数：
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64。其rank至少为2。
        - **updates** (Tensor) - 指定与 `self` 相加操作的Tensor，其数据类型与 `self` 相同。并且其shape应等于 :math:`indices.shape[:-1] + input\_x.shape[indices.shape[-1]:]` 。

    返回：
        Tensor，shape和数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型不为int32或int64。
        - **ValueError** - `self` 的rank小于 `indices.shape` 的最后一维。
        - **RuntimeError** - 在CPU平台中，`indices` 中的值超出了 `self` 的索引范围。
