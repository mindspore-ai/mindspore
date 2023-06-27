mindspore.ops.tensor_scatter_elements
=======================================

.. py:function:: mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis=0, reduction="none")

    将 `updates` 中所有的元素按照 `reduction` 指定的归约操作写入 `input_x` 中 `indices` 指定的索引处。
    `axis` 控制scatter操作的方向。
    `input_x` 、 `indices` 、 `updates` 三者的rank都必须大于或等于1。

    下面看一个三维的例子：

    .. code-block::

        output[indices[i][j][k]][j][k] = updates[i][j][k]  # if axis == 0, reduction == "none"

        output[i][indices[i][j][k]][k] += updates[i][j][k]  # if axis == 1, reduction == "add"

        output[i][j][indices[i][j][k]] = updates[i][j][k]  # if axis == 2, reduction == "none"

    .. warning::
        - 如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。
        - 在Ascend平台上，目前仅支持 `reduction` 设置为 ``"none"`` 的实现。
        - 在Ascend平台上，`input_x` 仅支持float16和float32两种数据类型。

    .. note::
        如果 `indices` 的值超出 `input_x` 索引上下界，则相应的 `updates` 不会更新到 `input_x` ，也不会抛出索引错误。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input_x** (Tensor) - 输入Tensor。rank必须大于等于1。
        - **indices** (Tensor) - `input_x` 要进行scatter操作的目标索引。数据类型为int32或int64，rank必须和 `input_x` 一致，取值范围是[-s, s)，s是 `input_x` 在 `axis` 指定轴的size。
        - **updates** (Tensor) - 指定与 `input_x` 进行scatter操作的Tensor，其数据类型与 `input_x` 类型相同，shape与 `indices` 的shape相同。
        - **axis** (int) - `input_x` 执行scatter操作的轴。取值范围是[-r, r)，其中r是 `input_x` 的rank。默认值: ``0`` 。
        - **reduction** (str) - 指定进行的规约操作。支持 ``"none"`` ， ``"add"`` 。默认值： ``"none"`` 。当 `reduction` 设置为 ``"none"`` 时，`updates` 将根据 `indices` 赋值给 `input_x`。当 `reduction` 设置为 ``"add"`` 时，`updates` 将根据 `indices` 累加到 `input_x`。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型不满足int32或int64。
        - **ValueError** - `input_x` 、 `indices` 和 `updates` 中，任意一者的rank小于1。
        - **ValueError** - `updates` 的shape和 `indices` 的shape不一致。
        - **ValueError** - `updates` 的rank和 `input_x` 的rank不一致。
        - **RuntimeError** - `input_x` 的数据类型和 `updates` 的数据类型不能隐式转换。