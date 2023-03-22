mindspore.ops.amax
==================

.. py:function:: mindspore.ops.amax(input, axis=None, keepdims=False, *, initial=None, where=None)

    默认情况下，移除输入所有维度，返回 `input` 中的最大值。也可仅缩小指定维度 `axis` 大小至1。 `keepdims` 控制输出和输入的维度是否相同。

    参数：
        - **input** (Tensor[Number]) - 输入Tensor，其数据类型为数值型。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。
        - **axis** (Union[int, tuple(int), list(int)]) - 要减少的维度。默认值: None，缩小所有维度。只允许常量值。假设 `input` 的秩为r，取值范围[-r,r)。
        - **keepdims** (bool) - 如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值：False。

    关键字参数：
        - **initial** (scalar, 可选) - 输出元素的最大值。如果 `input` 为空，则该参数必须设置。默认值：None。
        - **where** (Tensor[bool], 可选) - 表示是否需要将 `input` 中的原始值替换为 `initial` 值的Tensor。若为True则不替换，若为False则替换。`where` 中为False的位置，必须提供对应的 `initial` 值。默认值：True。

    返回：
        Tensor。

        - 如果 `axis` 为None，且 `keepdims` 为False，则输出一个零维Tensor，表示输入Tensor中所有元素的最大值。
        - 如果 `axis` 为int，取值为1，并且 `keepdims` 为False，则输出的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keepdims` 为False，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、tuple或list。
        - **TypeError** - `keepdims` 不是bool类型。
        - **ValueError** - `axis` 超出范围。
