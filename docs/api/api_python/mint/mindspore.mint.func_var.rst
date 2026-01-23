mindspore.mint.var
==================

.. py:function:: mindspore.mint.var(input, dim=None, *, correction=1, keepdim=False)

    计算指定维度 `dim` 上的方差。 `dim` 可以是单个维度、维度列表，也可以是 `None` ， 表示移除所有维度。

    方差 (:math:`\delta ^2`) 计算如下：

    .. math::
        \delta ^2 = \frac{1}{\max(0, N - \delta N)}\sum^{N - 1}_{i = 0}(x_i - \bar{x})^2

    其中 :math:`x` 表示用来计算方差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量，:math:`\delta N` 则为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 用来求方差的Tensor。
        - **dim** (None，int，tuple(int), 可选) - 用来进行归约计算的维度。默认值为 ``None`` ，所有维度都进行归约计算。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认为Bessel校正，默认值为 ``1`` 。
        - **keepdim** (bool, 可选) - 是否保留输出Tensor的维度。如果为 ``True`` ，则保留缩小的维度，其大小为1，否则移除维度。默认值为 ``False`` 。

    返回：
        Tensor，方差。
        假设输入 `input` 的shape为 :math:`(x_0, x_1, ..., x_R)` ：

        - 如果 `dim` 为()，且 `keepdim` 为 ``False`` ，则返回一个零维Tensor，表示输入Tensor `input` 中所有元素的方差。
        - 如果 `dim` 为int，如取值为 ``1`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `dim` 为tuple(int)或者list(int)，如取值为 ``(1, 2)`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不是bfloat16，float16或flaot32。
        - **TypeError** - 如果 `dim` 不是None，int，list或tuple类型。
        - **TypeError** - 如果 `correction` 不是int类型。
        - **TypeError** - 如果 `keepdim` 不是bool类型。
        - **ValueError** - 如果 `dim` 不在 :math:`[-input.ndim, input.ndim)` 范围内。