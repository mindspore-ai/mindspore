mindspore.ops.Tril
===================

.. py:class:: mindspore.ops.Tril(diagonal=0)

    返回单个或一批二维矩阵下三角形部分，其他位置的元素将被置零。
    矩阵的下三角形部分定义为对角线本身和对角线以下的元素。

    参数：
        - **diagonal** (int，可选) - 指定对角线位置，默认值：0，指定主对角线。
    输入：
        - **x** (Tensor) - 输入Tensor。shape为 :math:`(x_1, x_2, ..., x_R)` ，其rank至少为2。
          支持的数据类型有包括所有数值型和bool类型。

    输出：
        Tensor，其数据类型和shape维度与 `input_x` 相同。shape的第一个维度等于 `segment_ids` 最后一个元素的值加1，其他维度与 `input_x` 一致。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `diagonal` 不是int类型。
        - **TypeError** - 如果 `x` 的数据类型既不是数值型也不是bool。
        - **ValueError** - 如果 `x` 的秩小于2。
