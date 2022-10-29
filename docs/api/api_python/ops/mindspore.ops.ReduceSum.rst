mindspore.ops.ReduceSum
=========================

.. py:class:: mindspore.ops.ReduceSum(keep_dims=False, skip_mode=False)

    默认情况下，输出Tensor各维度上的和，以达到对所有维度进行归约的目的。也可以对指定维度进行求和归约。

    通过指定 `keep_dims` 参数，来控制输出和输入的维度是否相同。

    参数：
        - **keep_dims** (bool) - 如果为True，则保留计算维度，长度为1。如果为False，则不保留计算维度。默认值：False，输出结果会降低维度。
        - **skip_mode** (bool) - 如果为True，并且axis为空tuple或空list，不进行ReduceSum计算,axis为其他值，正常运算。如果为False，则正常进行运算。默认值：False。

    输入：
        - **x** (Tensor[Number]) - ReduceSum的输入，其数据类型为Number。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。
        - **axis** (Union[int, tuple(int), list(int)]) - 要减少的维度。默认值: ()，当skip_mode为False时，缩小所有维度。只允许常量值，取值范围[-rank(`x`), rank(`x`))。

    输出：
        Tensor，具有与输入 `x` 相同的shape。

        - 如果轴为()，且keep_dims为False，skip_mode为False，则输出一个0维Tensor，表示输入Tensor中所有元素的和。

        - 如果轴为()，且skip_mode为True，则不进行ReduceSum运算，输出Tensor等于输入Tensor。

        - 如果轴为int，取值为2，并且keep_dims为False，则输出的shape为 :math:`(x_1, x_3, ..., x_R)` 。

        - 如果轴为tuple(int)或list(int)，取值为(2, 3)，并且keep_dims为False，则输出的shape为 :math:`(x_1, x_4, ..., x_R)` 。

    异常：
        - **TypeError** - `keep_dims` 不是bool。
        - **TypeError** - `skip_mode` 不是bool。
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `axis` 取值为None。