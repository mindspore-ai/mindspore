mindspore.ops.tril
===================

.. py:function:: mindspore.ops.tril(input, diagonal=0)

    返回输入Tensor `input` 的下三角形部分(包含对角线和下面的元素)，并将其他元素设置为0。

    参数：
        - **input** (Tensor) - 输入Tensor。shape为 :math:`(x_1, x_2, ..., x_R)` ，其rank至少为2。
          支持的数据类型有包括所有数值型和bool类型。
        - **diagonal** (int，可选) - 指定对角线位置，默认值：0，指定主对角线。

    返回：
        Tensor，其数据类型和shape维度与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `diagonal` 不是int类型。
        - **TypeError** - 如果 `input` 的数据类型既不是数值型也不是bool。
        - **ValueError** - 如果 `input` 的秩小于2。
