mindspore.ops.intopk
====================

.. py:function:: mindspore.ops.intopk(x1, x2, k)

    判断目标标签是否在前 `k` 个预测中。

    参数：
        - **x1** (Tensor) - 二维Tensor，对样本的预测。数据类型支持float16或float32。
        - **x2** (Tensor) - 一维Tensor，样本的标签。数据类型为int32。 `x2` 的大小必须与 `x1` 第一维度的大小相同。 `x2` 取值不可为负且必须小于或等于 `x1` 第二维度的大小。
        - **k** (int) - 指定在最后一维上参与比较的top元素的数量。

    返回：
        一维的bool类型Tensor，与 `x2` shape相同。对于 `x2` 中的样本标签 `i`，如果它在 `x1` 的前 `k` 个预测值中，则输出值为True，否则为False。

    异常：
        - **TypeError** - `k` 不是int类型。
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
        - **TypeError** - `x1` 的数据类型非float16或float32。
