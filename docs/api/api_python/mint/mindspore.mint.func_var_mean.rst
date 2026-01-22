mindspore.mint.var_mean
=======================

.. py:function:: mindspore.mint.var_mean(input, dim=None, *, correction=1, keepdim=False)

    计算tensor在指定维度上的方差及平均值。

    方差 (:math:`\sigma ^2`) 计算如下：

    .. math::
        \sigma ^2 = \frac{1}{N - \delta N} \sum_{j=0}^{N-1} \left(self_{ij} - \overline{x_{i}}\right)^{2}

    其中 :math:`x` 表示用来计算方差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量， :math:`\delta N` 为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int)], 可选) - 指定计算维度。默认 ``None``。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认采用贝塞尔校正，取值 ``1`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(var, mean)。