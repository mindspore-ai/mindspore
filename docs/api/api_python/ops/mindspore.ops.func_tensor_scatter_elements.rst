mindspore.ops.tensor_scatter_elements
=======================================

.. py:function:: mindspore.ops.tensor_scatter_elements(input_x, indices, updates, axis=0, reduction="none")

    返回一个新tensor，根据指定索引和更新值对 `input_x` 中的元素进行指定操作（替换或相加）。

    不支持隐式类型转换。

    举例：一个三维输入tensor的返回为

    .. code-block::

        output[indices[i][j][k]][j][k] = updates[i][j][k]  # if axis == 0, reduction == "none"

        output[i][indices[i][j][k]][k] += updates[i][j][k]  # if axis == 1, reduction == "add"

        output[i][j][indices[i][j][k]] = updates[i][j][k]  # if axis == 2, reduction == "none"

    .. warning::
        - 如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。
        - 在Ascend平台上，目前仅支持 `reduction` 设置为 ``"none"`` 的实现。
        - 在Ascend平台上，`input_x` 仅支持float16和float32两种数据类型。
        - 该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        如果 `indices` 的值超出 `input_x` 索引上下界，则相应的 `updates` 不会更新到 `input_x` ，也不会抛出索引错误。
        仅当indices的shape和updates的shape相同时支持求反向梯度。

    参数：
        - **input_x** (Tensor) - 输入tensor，秩大于等于1。
        - **indices** (Tensor) - 指定索引。
        - **updates** (Tensor) - 更新值。
        - **axis** (int) - 指定索引所在的轴。默认 ``0`` 。
        - **reduction** (str) - 指定操作。支持 ``none`` ， ``add`` 。默认 ``none`` 。

          - 如果为 ``none`` ，将把 `updates` 中的元素赋值给 `input_x`。
          - 如果为 ``add`` ，将把 `updates` 中的元素与 `input_x` 中的元素相加。

    返回：
        Tensor
