mindspore.ops.vander
====================

.. py:function:: mindspore.ops.vander(x, N=None)

    生成一个范德蒙矩阵。
    返回矩阵的各列是入参的幂。第 i 个输出列是输入向量元素的幂，其幂为 :math:`N-i-1`。

    参数：
        - **x** (Tensor) - 1-D 输入阵列。
        - **N** (int，可选) - 返回矩阵的列数。默认值： ``None`` ，默认为 :math:`len(x)`。

    返回：
        Tensor，矩阵的列为 :math:`x^0, x^1, ..., x^{(N-1)}`。

    异常：
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **ValueError** - 如果 `x` 不是1-D。
        - **TypeError** - 如果 `N` 不是 int。
        - **ValueError** - 如果 `N` <= 0。
