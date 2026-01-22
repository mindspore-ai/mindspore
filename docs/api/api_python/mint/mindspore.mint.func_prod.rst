mindspore.mint.prod
===================

.. py:function:: mindspore.mint.prod(input, *, dtype=None) -> Tensor

    计算tensor所有元素的乘积。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor。

    .. py:function:: mindspore.mint.prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    默认情况下，使用指定维度的所有元素的乘积代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 `keepdim` 控制输出和输入的维度是否相同。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。
        - **dim** (int) - 指定计算维度。
        - **keepdim** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor。

        - 如果 `dim` 为int，取值为1，并且 `keepdim` 为 ``False`` ，则输出的shape为 :math:`(input_0, input_2, ..., input_R)` 。

    异常：
        - **ValueError** - `dim` 超出范围。
