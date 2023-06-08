mindspore.ops.ReduceAny
=========================

.. py:class:: mindspore.ops.ReduceAny(keep_dims=False)

    默认情况下，通过对指定维度所有元素进行逻辑或运算来移除该维度。也可仅缩小该维度大小至1。 `keep_dims` 控制输出和输入的维度是否相同。

    参数：
        - **keep_dims** (bool) - 如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    输入：
        - **x** (Tensor[bool]) - bool类型的输入Tensor。
        - **axis** (Union[int, tuple(int), list(int)]) - 要规约计算的维度。默认值:  ``()`` ，在所有维度上规约。只允许常量值。取值必须在[-rank(`x`), rank(`x`))范围内。

    输出：
        bool类型的Tensor。

        - 如果 `axis` 为()，且 `keep_dims` 为 ``False`` ，
          则输出一个零维Tensor，表示输入Tensor中所有元素的逻辑或运算结果。
        - 如果 `axis` 为int，值为2，并且 `keep_dims` 为 ``False`` ，
          则输出Tensor的shape为： :math:`(x_1, x_3, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)，值为(2, 3)，并且 `keep_dims` 为 ``False`` ，
          则输出Tensor的shape为： :math:`(x_1, x_4, ..., x_R)` 。

    异常：
        - **TypeError** - `keep_dims` 不是bool类型。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、Tuple或List。
