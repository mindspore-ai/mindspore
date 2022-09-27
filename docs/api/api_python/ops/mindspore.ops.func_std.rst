mindspore.ops.std
==================

.. py:function:: mindspore.ops.std(input_x, axis=(), unbiased=True, keep_dims=False)

    默认情况下，输出Tensor各维度上的标准差与均值，也可以对指定维度求标准差与均值。如果 `axis` 是维度列表，则减少对应的维度。

    参数：
        - **input_x** (Tensor[Number]) - 输入Tensor，其数据类型为数值型。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。
        - **axis** (Union[int, tuple(int), list(int)]) - 要减少的维度。默认值: ()，缩小所有维度。只允许常量值。假设 `x` 的秩为r，取值范围[-r,r)。
        - **unbiased** (bool) - 如果为True，使用贝塞尔校正。否则不使用贝塞尔校正。默认值：True。
        - **keep_dims** (bool) - 如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值：False。

    返回：
        Tensor。

        - 如果 `axis` 为()，且 `keep_dims` 为False，则输出一个0维Tensor，表示输入Tensor中所有元素的标准差。
        - 如果 `axis` 为int，取值为1，并且 `keep_dims` 为False，则输出的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keep_dims` 为False，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、Tuple或List。
        - **TypeError** - `keep_dims` 不是bool类型。
        - **ValueError** - `axis` 超出范围。
