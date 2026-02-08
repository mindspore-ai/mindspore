mindspore.mint.var
==================

.. py:function:: mindspore.mint.var(input, dim=None, *, correction=1, keepdim=False)

    计算tensor在指定维度上的方差。

    方差 (:math:`\delta ^2`) 计算如下：

    .. math::
        \delta ^2 = \frac{1}{\max(0, N - \delta N)}\sum^{N - 1}_{i = 0}(x_i - \bar{x})^2

    其中 :math:`x` 表示用来计算方差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量，:math:`\delta N` 则为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (None，int，tuple(int), 可选) - 指定计算维度。如果为 ``None`` ，计算 input 中的所有元素。默认 ``None`` 。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认采用贝塞尔校正，取值 ``1`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor