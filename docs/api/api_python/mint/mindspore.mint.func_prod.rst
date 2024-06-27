mindspore.mint.prod
===================

.. py:function:: mindspore.mint.prod(input, dim=None, keepdim=False, *, dtype=None)

    默认情况下，使用指定维度的所有元素的乘积代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 `keepdim` 控制输出和输入的维度是否相同。

    参数：
        - **input** (Tensor[Number]) - 输入Tensor，其数据类型为数值型。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **dim** (int) - 要减少的维度。默认值:  ``None`` ，缩小所有维度。只允许常量值。假设 `input` 的秩为r，取值范围[-r,r)。
        - **keepdim** (bool) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 期望输出Tensor的类型。默认值： ``None`` 。

    返回：
        Tensor。

        - 如果 `dim` 为 ``None`` ，且 `keepdim` 为 ``False`` ，则输出一个零维Tensor，表示输入Tensor中所有元素的乘积。
        - 如果 `dim` 为int，取值为1，并且 `keepdim` 为 ``False`` ，则输出的shape为 :math:`(input_0, input_2, ..., input_R)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `dim` 不是int。
        - **TypeError** - `keepdim` 不是bool类型。
        - **ValueError** - `dim` 超出范围。
