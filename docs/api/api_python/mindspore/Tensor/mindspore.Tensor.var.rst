mindspore.Tensor.var
====================

.. py:method:: mindspore.Tensor.var(axis=None, ddof=0, keepdims=False) -> Tensor

    在指定维度上的方差。

    方差是平均值的平方偏差的平均值，即： :math:`var = mean(abs(x - x.mean())**2)` 。

    返回方差值，默认情况下计算展开Tensor的方差，否则在指定维度上计算。

    .. note::
        不支持NumPy参数 `dtype` 、 `out` 和 `where` 。

    参数：
        - **axis** (Union[None, int, tuple(int)]，可选) - 维度，在指定维度上计算方差。当 `axis=None` 时计算展开Tensor的方差。默认值： ``None`` 。
        - **ddof** (int，可选) - δ自由度。默认值： ``0`` 。计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。
        - **keepdims** (bool，可选) - 是否保留输出Tensor的维度。如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认值： ``False`` 。

    返回：
        含有方差值的Tensor。

    异常：
        - **TypeError** - 如果 `axis` 不是None，int或tuple类型。
        - **TypeError** - 如果 `ddof` 不是int类型。
        - **TypeError** - 如果 `keepdims` 不是bool类型。
        - **ValueError** - 如果 `axis` 不在 :math:`[-self.ndim, self.ndim)` 范围内。

    .. seealso::
        - :func:`mindspore.Tensor.mean` ：通过对Tensor中的所有元素求平均值来减少Tensor的维数。
        - :func:`mindspore.Tensor.std` ：计算沿指定轴的标准差。

    .. py:method:: mindspore.Tensor.var(dim=None, *, correction=1, keepdim=False) -> Tensor
        :noindex:

    计算指定维度 `dim` 上的方差。 `dim` 可以是单个维度、维度列表，也可以是 `None` ，表示移除所有维度。

    方差 (:math:`\delta ^2`) 计算如下：

    .. math::
        \delta ^2 = \frac{1}{\max(0, N - \delta N)}\sum^{N - 1}_{i = 0}(x_i - \bar{x})^2

    其中 :math:`x` 表示用来计算方差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量， :math:`\delta N` 则为 `correction` 的值。

    参数：
        - **dim** (None，int，tuple(int)，可选) - 用来进行归约计算的维度。默认值为 ``None`` ，所有维度都进行归约计算。

    关键字参数：
        - **correction** (int，可选) - 样本大小和样本自由度之间的差异。 `correction=1` 则对应Bessel校正，默认值为 ``1`` 。
        - **keepdim** (bool，可选) - 是否保留输出Tensor的维度。如果为 ``True`` ，则保留缩小的维度，其大小为1，否则移除维度。默认值为 ``False`` 。

    返回：
        Tensor，方差。
        假设 `self` Tensor的shape为 :math:`(x_0, x_1, ..., x_R)` ：

        - 如果 `dim` 为()，且 `keepdim` 为 ``False`` ，则返回一个零维Tensor，表示 `self` Tensor中所有元素的方差。
        - 如果 `dim` 为int，如取值为 ``1`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_2, ..., x_R)` 。
        - 如果 `dim` 为tuple(int)或list(int)，如取值为 ``(1, 2)`` ，且 `keepdim` 为 ``False`` ，则返回Tensor的shape为 :math:`(x_0, x_3, ..., x_R)` 。

    异常：
        - **TypeError** - 如果 `dim` 不是None、int、list或tuple类型。
        - **TypeError** - 如果 `correction` 不是int类型。
        - **TypeError** - 如果 `keepdim` 不是bool类型。
        - **ValueError** - 如果 `dim` 不在 :math:`[-self.ndim, self.ndim)` 范围内。