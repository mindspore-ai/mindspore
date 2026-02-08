mindspore.Tensor.std
====================

.. py:method:: mindspore.Tensor.std(axis=None, ddof=0, keepdims=False) -> Tensor

    详情请参考 :func:`mindspore.ops.std`。

    .. py:method:: mindspore.Tensor.std(dim=None, *, correction=1, keepdim=False) -> Tensor
        :noindex:

    计算指定维度 `dim` 上的标准差。 `dim` 可以是单个维度或维度列表，也可以是 `None` ，表示移除所有维度。

    标准差 (:math:`\sigma`) 计算如下：

    .. math::
        \sigma =\sqrt{\frac{1}{N-\delta N}\sum_{j-1}^{N-1}\left(self_{ij}-\overline{x_{i}}\right)^{2}}

    其中 :math:`x` 表示用来计算标准差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量， :math:`\delta N` 则为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **dim** (None, int, tuple(int), 可选) - 用来进行归约计算的维度。默认值为 ``None`` ，所有维度都进行归约计算。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认值为 ``1`` ，即对应Bessel校正。
        - **keepdim** (bool, 可选) - 是否保留输出Tensor的维度。如果为 ``True`` ，则保留缩小的维度，其大小为1，否则移除维度。默认值为 ``False`` 。

    返回：
        Tensor，标准差。

        假设输入 `self` 的shape为 :math:`(x_0, x_1, ..., x_R)` ：

        - 如果 `dim` 为()，且 `keepdim` 为 ``False`` ，则返回一个零维Tensor，表示输入Tensor `self` 中所有元素的标准差。
        - 如果 `dim` 为int，如取值为 ``1`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `dim` 为tuple(int)或者list(int)，如取值为 ``(1, 2)`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
        - **TypeError** - 如果 `self` 的数据类型不是bfloat16、float16或float32。
        - **TypeError** - 如果 `dim` 不是None、int或tuple类型。
        - **TypeError** - 如果 `correction` 不是int类型。
        - **TypeError** - 如果 `keepdim` 不是bool类型。
        - **ValueError** - 如果 `dim` 不在 :math:`[-self.ndim, self.ndim)` 范围内。

