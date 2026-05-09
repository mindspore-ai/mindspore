mindspore.numpy.array_split
=================================

.. py:function:: mindspore.numpy.array_split(x, indices_or_sections, axis=0)

    将一个Tensor切分为多个Sub-Tensors。

    .. note::
        目前， ``array_split`` 仅支持CPU上的 ``mindspore.float32`` 。

    ``np.split`` 和 ``np.array_split`` 之间的唯一区别是， ``np.array_split`` 允许 ``indices_or_sections`` 是一个不用等分 ``axis`` 的整数。对于长度为l的Tensor，其应当被分割成n个部分，返回的子数组中一部分子数组shape为 :math:`l//n+1` ，剩余子数组shape为 :math:`l//n` 。

    参数：
        - **x** (Tensor) - 待切分的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 如果是整数 :math:`N` ，Tensor将沿指定的轴被拆分成 :math:`N` 个子张量。如果是tuple(int)、list(int)或排序后的整数序列，则这些条目表示在指定轴上的拆分位置。例如，给定 :math:`[2,3]` 在 :math:`axis=0` 上拆分，则结果将是三个Sub-Tensors，为 :math:`x[:2]` ， :math:`x[2:3]` ， :math:`x[3:]` 。如果某个索引超出了指定轴的维度，相应地将返回一个空子数组。
        - **axis** (int) - 要沿其切分的轴。默认值： ``0`` 。

    返回：
        Sub-Tensors列表。

    异常：
        - **TypeError** - 如果参数 ``indices_or_sections`` 不是整数、tuple(int)或list(int)，或者参数 ``axis`` 不是整数。
        - **ValueError** - 如果参数 ``axis`` 超出 :math:`[-x.ndim,x.ndim)` 的范围。