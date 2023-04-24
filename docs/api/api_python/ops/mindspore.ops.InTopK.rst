mindspore.ops.InTopK
====================

.. py:class:: mindspore.ops.InTopK(k)

    判断目标标签是否在前 `k` 个预测中。

    更多参考详见 :func:`mindspore.ops.intopk`。

    参数：
        - **k** (int) - 指定在最后一维上参与比较的top元素的数量。

    输入：
        - **x1** (Tensor) - 二维Tensor，对样本的预测。数据类型支持float16或float32。
        - **x2** (Tensor) - 一维Tensor，样本的标签。数据类型为int32。 `x2` 的大小必须与 `x1` 第一维度的大小相同。 `x2` 取值不可为负且必须小于或等于 `x1` 第二维度的大小。

    输出：
        一维的bool类型Tensor，与 `x2` shape相同。对于 `x2` 中的样本标签 `i`，如果它在 `x1` 的前 `k` 个预测值中，则输出值为 ``True`` ，否则为 ``False`` 。
