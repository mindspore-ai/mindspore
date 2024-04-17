mindspore.ops.ReduceMax
========================

.. py:class:: mindspore.ops.ReduceMax(keep_dims=False)

    默认情况下，使用指定维度的最大值代替该维度的其他元素，以移除该维度。也可仅缩小该维度大小至1。 `keep_dims` 控制输出和输入的维度是否相同。

    通过指定 `keep_dims` 参数，来控制输出和输入的维度是否相同。

    .. note::
        Tensor类型的 `axis` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **keep_dims** (bool) - 如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    输入：
        - **x** (Tensor[Number]) - 输入Tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor]) - 要进行规约计算的维度。默认值： ``()`` ，在所有维度上进行规约。只允许常量值。假设 `x` 的秩为r，取值范围[-r,r)。

    输出：
        与输入 `x` 具有相同数据类型的Tensor。

        - 如果 `axis` 为 ``()`` ，且 `keep_dims` 为 ``False`` ，则输出一个零维Tensor，表示输入Tensor中所有元素的最大值。
        - 如果 `axis` 为int，取值为1，并且 `keep_dims` 为 ``False`` ，则输出的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，取值为(1, 2)，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。
        - 如果 `axis` 为一维Tensor，取值为[1, 2]，并且 `keep_dims` 为 ``False`` ，则输出Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - `keep_dims` 不是bool类型。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是以下数据类型之一：int、Tuple、List或Tensor。
        - **ValueError** - `axis` 超出范围。
