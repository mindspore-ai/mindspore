mindspore.Tensor.transpose
==========================

.. py:method:: mindspore.Tensor.transpose(*axes)

    返回被转置后的Tensor。

    - 对于一维Tensor，这没有影响，因为转置后的向量是相同的。
    - 对于二维Tensor，是标准的矩阵转置。
    - 对于n维Tensor，如果提供了维度，则它们的顺序代表维度的置换方式。

    如果未提供轴，且Tensor.shape等于(i[0], i[1],...i[n-2], i[n-1])，则Tensor.transpose().shape等于(i[n-1], i[n-2], ... i[1], i[0])。

    参数：
        - **axes** (Union[None, tuple(int), list(int), int], 可选) - 如果 `axes` 为None或未设置，则该方法将反转维度。如果 `axes` 为tuple(int)或list(int)，则Tensor.transpose()把Tensor转置为新的维度。如果 `axes` 为整数，则此表单仅作为元组/列表表单的备选。

    返回：
        Tensor，具有与输入Tensor相同的维度，其中维度被准确的排列。

    异常：
        - **TypeError** - 输入参数类型有误。
        - **ValueError** - `axes` 的数量不等于Tensor.ndim。