mindspore.ops.std
==================

.. py:function:: mindspore.ops.std(input, axis=None, ddof=0, keepdims=False)

    默认情况下，输出Tensor各维度上的标准差，也可以对指定维度求标准差。如果 `axis` 是维度列表，则计算对应维度的标准差。

    .. note::
        如果 `ddof` 是0、1、True或False，支持的平台只有 `Ascend` 和 `CPU` 。其他情况下，支持平台是 `Ascend` 、 `GPU` 和 `CPU` 。

    参数：
        - **input** (Tensor[Number]) - 输入Tensor，其数据类型为数值型。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 要减少的维度。只允许常量值。假设 `input` 的秩为r，取值范围[-r,r)。默认值: None，缩小所有维度。
        - **ddof** (Union[int, bool]，可选) - δ自由度。如果为整数，计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。如果为True，使用Bessel校正。如果是False，使用偏置估计来计算标准差。默认值：0。
        - **keepdims** (bool，可选) - 是否保留输出Tensor的维度。如果为True，则保留缩小的维度，大小为1。否则移除维度。默认值：False。
 
    返回：
        Tensor，标准差。
        假设输入 `input` 的shape为 :math:`(x_0, x_1, ..., x_R)` ：
        - 如果 `axis` 为()，且 `keepdims` 为False，则输出一个零维Tensor，表示输入Tensor `input` 中所有元素的标准差。
        - 如果 `axis` 为int，取值为1，并且 `keepdims` 为False，则输出的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keepdims` 为False，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：None、int、tuple或list。
        - **TypeError** - `keepdims` 不是bool类型。
        - **ValueError** - `axis` 超出范围。
