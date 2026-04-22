mindspore.ops.vander
====================

.. py:function:: mindspore.ops.vander(x, N=None)

    生成一个范德蒙矩阵。
    返回矩阵的各列是入参各元素的幂。第 i 个输出列是输入向量元素的幂，其幂为 :math:`N - i - 1`。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `increasing` ，接口签名将变更为
        ``mindspore.ops.vander(x, N=None, increasing=False)`` 。
        该变更与当前版本兼容，不使用 `increasing` 参数时行为不变。

    参数：
        - **x** (Tensor) - 一维输入tensor。
        - **N** (int，可选) - 返回矩阵的列数。默认 ``None`` ，为 :math:`len(x)`。

    返回：
        Tensor
