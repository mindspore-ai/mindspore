mindspore.mint.scatter_add
==========================

.. py:function:: mindspore.mint.scatter_add(input, dim, index, src)

    将 `src` 中所有的元素添加到 `input` 中 `index` 指定的索引处。
    其中 `dim` 控制scatter_add操作的轴。
    `input` 、 `index` 、 `src` 三者的rank都必须大于或等于1。

    下面看一个三维的例子：

    .. code-block::

        input[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0

        input[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1

        input[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

    参数：
        - **input** (Tensor) - 输入Tensor。rank必须大于等于1。
        - **dim** (int) - `input` 执行scatter_add操作的轴。取值范围是[-r, r)，其中r是 `input` 的rank。默认值: ``0`` 。
        - **index** (Tensor) - `input` 要进行scatter_add操作的目标索引。数据类型为int32或int64，rank必须和 `input` 一致。除了 `dim` 指定的维度， `index` 的每一维的size都需要小于等于 `input` 对应维度的size。
        - **src** (Tensor) - 指定与 `input` 进行scatter_add操作的Tensor，其数据类型与 `input` 类型相同，shape中每一维的size必须大于等于 `index` 。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `index` 的数据类型不满足int32或int64。
        - **ValueError** - `input` 、 `index` 和 `src` 中，任意一者的rank小于1。
        - **ValueError** - `src` 的shape和 `index` 的shape不一致。
        - **ValueError** - `src` 的rank和 `input` 的rank不一致。
