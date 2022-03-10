mindspore.ops.ReduceMax
========================

.. py:class:: mindspore.ops.ReduceMax(keep_dims=False)

    默认情况下，输出张量各维度上的最大值，以达到对所有维度进行归约的目的。也可以对指定维度进行求最大值归约。

    通过指定 `keep_dims` 参数，来控制输出和输入的维度是否相同。

    **参数：**

    - **keep_dims** (bool) - 如果为True，则保留计算的维度，长度为1。如果为False，则不保留计算维度。默认值：False，输出结果会降低维度。

    **输入：**

    - **x** (Tensor[Number]) - ReduceMax的输入，任意维度的Tensor，秩应小于8。其数据类型为number。
    - **axis** (Union[int, tuple(int), list(int)]) - 指定计算维度。默认值：()，即计算所有元素的最大值。只允许常量值，取值范围[-rank(x), rank(x))。

    **输出：**

    Tensor，shape与输入 `x` 相同。

    - 如果轴为()，且keep_dims为False，则输出一个0维Tensor，表示输入Tensor中所有元素的最大值。

    - 如果轴为int，取值为2，并且keep_dims为False，则输出的shape为 :math:`(x_1, x_3, ..., x_R)` 。

    - 如果轴为tuple(int)，取值为(2, 3)，并且keep_dims为False，则输出的shape为 :math:`(x_1, x_4, ..., x_R)` 。

    **异常：**

    - **TypeError** - `keep_dims` 不是bool。
    - **TypeError** - `x` 不是tensor。
    - **ValueError** - `axis` 不是int、tuple或list。