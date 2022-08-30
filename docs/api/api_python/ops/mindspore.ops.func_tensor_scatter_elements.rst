mindspore.ops.tensor_scatter_elements
=======================================

.. py:function:: mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis=0, reduction="none")

    根据索引逐元素更新输入Tensor的值。
    接受三个输入， `input_x`、`indices` 和 `updates` 。三者的秩都大于等于1。`input_x` 中的数据和 `updates` 中的数据会按照 `indices` 提取出来
    做 `reduction` 指定的操作并更新结果到输出。下面看一个三维的例子：

    .. code-block::

        output[indices[i][j][k]][j][k] = updates[i][j][k]  # if axis == 0, reduction == "none"

        output[i][indices[i][j][k]][k] += updates[i][j][k]  # if axis == 1, reduction == "add"

        output[i][j][indices[i][j][k]] = updates[i][j][k]  # if axis == 2, reduction == "none"

    .. warning::
        - 如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。
        - 在Ascend平台上，目前仅支持 `reduction` 设置为"none"的实现。
        - 在Ascend平台上，`input_x` 仅支持float16和float32两种数据类型。

    .. note::
        如果 `indices` 的某些值超出范围，则相应的 `updates` 不会更新到 `input_x` ，也不会抛出索引错误。

    参数：
        - **input_x** (Tensor) - 输入Tensor。 `input_x` 其rank必须至少为1。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64的。其rank必须和 `input_x` 一致。取值范围是[-s, s)，这里的s是 `input_x` 在 `axis` 指定轴的size。
        - **updates** (Tensor) - 指定与 `input_x` 进行reduction操作的Tensor，其数据类型与输入的数据类型相同。updates的shape必须等于indices的shape。
        - **axis** (int) - `input_x` reduction操作的轴，默认值是0。取值范围是[-r, r)，其中r是 `input_x` 的秩。
        - **reduction** (str) - 指定进行的reduction操作。默认值是"none"，可选"add"。

    返回：
        Tensor，shape和数据类型与输入 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input_x` 、 `indices` 和 `updates` 中，任意一者的秩小于1。
        - **ValueError** - `updates` 的shape和 `indices` 的shape不一样。
        - **ValueError** - `updates` 的秩和 `input_x` 的秩不一样。
        - **RuntimeError** - `input_x` 的数据类型和 `updates` 的数据类型不能隐式转换。