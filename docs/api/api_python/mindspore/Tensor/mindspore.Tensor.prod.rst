mindspore.Tensor.prod
=====================

.. py:method:: mindspore.Tensor.prod(dim=None, keepdim=False, dtype=None) -> Tensor

    默认情况下，使用指定维度的所有元素的乘积代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 `keepdim` 控制输出和输入的维度是否相同。

    参数：
        - **dim** (Union[int, tuple(int), list(int), Tensor]，可选) - 要减少的维度。默认值： ``None`` ，缩小所有维度。只允许常量值。假设 `self` 的秩为r，取值范围[-r,r)。
        - **keepdim** (bool，可选) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - 期望输出Tensor的类型。默认值： ``None`` 。

    返回：
        Tensor。

        - 如果 `dim` 为 ``None`` ，且 `keepdim` 为 ``False`` ，则输出一个零维Tensor，表示输入Tensor中所有元素的乘积。
        - 如果 `dim` 为int，取值为1，并且 `keepdim` 为 ``False`` ，则输出的shape为 :math:`(self_0, self_2, ..., self_R)` 。
        - 如果 `dim` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keepdim` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_0, self_3, ..., self_R)` 。
        - 如果 `dim` 为一维Tensor，例如取值为[1, 2]，并且 `keepdim` 为 ``False`` ，则输出Tensor的shape为 :math:`(self_0, self_3, ..., self_R)` 。

    异常：
        - **TypeError** - `dim` 不是以下数据类型之一：int、tuple、list或Tensor。
        - **TypeError** - `keepdim` 不是bool类型。
        - **ValueError** - `dim` 超出范围。

    .. py:method:: mindspore.Tensor.prod(axis=None, keep_dims=False, dtype=None) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.ops.prod`。
