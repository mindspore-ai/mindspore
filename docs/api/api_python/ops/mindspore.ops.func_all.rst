mindspore.ops.all
=================

.. py:function:: mindspore.ops.all(input, axis=None, keep_dims=False)

    默认情况下，通过对维度中所有元素进行“逻辑与”来减少 `input` 的维度。也可以沿轴减少 `input` 的维度。通过控制 `keep_dim` 来确定输出和输入的维度是否相同。

    参数：
        - **input** (Tensor[bool]) - 输入Tensor，其数据类型为bool型。shape是 :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。
        - **axis** (Union[int, tuple(int), list(int)], 可选) - 要减少的维度。只允许常量值。假设 `input` 的秩为r，取值范围[-r,r)。默认值：()，缩小所有维度。
        - **keep_dims** (bool, 可选) - 如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值：False。

    返回：
        Tensor，数据类型是bool。

        - 如果 `axis` 为None，且 `keep_dims` 为False，则输出一个零维Tensor，表示输入Tensor中所有元素进行“逻辑与”。
        - 如果 `axis` 为int，取值为2，并且 `keep_dims` 为False，则输出的shape为 :math:`(input_1, input_3, ..., input_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(2, 3)，并且 `keep_dims` 为False，则输出Tensor的shape为 :math:`(input_1, input_4, ..., input_R)` 。

    异常：
        - **TypeError** - `keep_dims` 不是bool类型。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、tuple或list。
