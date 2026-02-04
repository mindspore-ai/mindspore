mindspore.mint.std
==================

.. py:function:: mindspore.mint.std(input, dim=None, *, correction=1, keepdim=False)

    计算Tensor在指定维度上的标准差。

    标准差 (:math:`\sigma`) 计算如下：

    .. math::
        \sigma =\sqrt{\frac{1}{N-\delta N}\sum_{j-1}^{N-1}\left(s e l f_{i j}-\overline{x_{i}}\right)^{2}}

    其中 :math:`x` 表示用来计算标准差的样本集, :math:`\bar{x}` 表示样本的均值, :math:`N` 表示样本的数量，:math:`\delta N` 则为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (None, int, tuple(int), 可选) - 指定维度。如果为 ``None`` ，计算所有元素。默认 ``None`` 。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认为Bessel校正，默认 ``1`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
