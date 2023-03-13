mindspore.ops.ReduceStd
=======================

.. py:class:: mindspore.ops.ReduceStd(axis=(), unbiased=True, keep_dims=False)

    返回输入Tensor在 `axis` 维上每一行的标准差和均值。
    如果 `axis` 是维度列表，则减少列表内的所有维度。

    参数：
        - **axis** (Union[int, tuple(int), list(int)]，可选) - 要进行规约计算的维度。只允许常量值。假设 `input_x` 的秩为 `r` ，取值范围 :math:`[-r,r)` 。默认值: ()，对所有维度进行规约。
        - **unbiased** (bool，可选) - 是否应用贝塞尔校正。如果为True，则使用贝塞尔校正进行无偏估计。如果为False，则通过有偏估计计算标准差。默认值：False。
        - **keep_dims** (bool，可选) - 是否保持输入与输出Tensor维度一致。如果为True，保留 `axis` 指定的维度，但其尺寸变为1。如果为Fasle，不保留这些维度。默认值：False。

    输入：
        - **input_x** (Tensor[Number]) - 输入Tensor。其数据类型为Number，shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。

    输出：
        Tuple (output_std, output_mean)，分别为标准差和均值。

    异常：
        - **TypeError** - 如果 `keep_dims` 不是bool。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **ValueError** - 如果 `axis` 不是int、tuple或list。
