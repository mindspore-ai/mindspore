mindspore.nn.Norm
==================

.. py:class:: mindspore.nn.Norm(axis=(), keep_dims=False)

    计算向量的范数，目前包括欧几里得范数，即 :math:`L_2`-norm。

    .. math::
        norm(x) = \sqrt{\sum_{i=1}^{n} (x_i^2)}

    **参数：**

    - **axis** (Union[tuple, int]) - 指定计算向量范数的轴。默认值：()。
    - **keep_dims** (bool) - 如果为True，则 `axis` 中指定轴的维度大小为1。否则，`axis` 的维度将从输出shape中删除。默认值：False。

    **输入：**

    - **x** (Tensor) - 输入任意维度Tensor，不为空。数据类型应为float16或float32。

    **输出：**

    Tensor，输出为Tensor，如果'keep_dims'为True，则将保留'axis'指定的维度且为1；否则，将移除'axis'中指定的维度。数据类型与 `x` 相同。

    **异常：**

    - **TypeError** - `axis` 既不是int也不是tuple。
    - **TypeError** - `keep_dims` 不是bool。