mindspore.ops.ReduceStd
=======================

.. py:class:: mindspore.ops.ReduceStd(axis=(), unbiased=True, keep_dims=False)

    返回输入Tensor在 `axis` 对应维度上的标准差和均值。

    .. note::
        Tensor类型的 `axis` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **axis** (Union[int, tuple(int), list(int), Tensor]，可选) - 要进行规约计算的维度。只允许常量值。假设 `input_x` 的秩为 `r` ，取值范围 :math:`[-r,r)` 。默认值:  ``()`` ，对所有维度进行规约。
        - **unbiased** (bool，可选) - 是否应用贝塞尔校正。如果为 ``True`` ，则使用贝塞尔校正进行无偏估计。如果为 ``False`` ，则通过有偏估计计算标准差。默认值： ``False`` 。
        - **keep_dims** (bool，可选) - 是否保持输入与输出Tensor维度一致。如果为 ``True`` ，保留 `axis` 指定的维度，但其尺寸变为1。如果为 ``False`` ，不保留这些维度。默认值： ``False`` 。

    输入：
        - **input_x** (Tensor[Number]) - 输入Tensor。shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。支持的数据类型：float16、float32。

    输出：
        Tuple (output_std, output_mean)，分别为标准差和均值。

    异常：
        - **TypeError** - 如果 `keep_dims` 不是bool。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **ValueError** - 如果 `axis` 不是int、tuple、list或Tensor。
