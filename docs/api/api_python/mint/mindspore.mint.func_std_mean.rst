mindspore.mint.std_mean
=======================

.. py:function:: mindspore.mint.std_mean(input, dim=None, *, correction=1, keepdim=False)

    计算tensor在指定维度上的标准差及平均值。

    标准差 (:math:`\sigma`) 计算如下：

    .. math::
        \sigma = \sqrt{\frac{1}{N - \delta N} \sum_{j=0}^{N-1} \left(self_{ij} - \overline{x_{i}}\right)^{2}}

    其中 :math:`x` 表示用来计算标准差的样本集， :math:`\bar{x}` 表示样本的均值， :math:`N` 表示样本的数量， :math:`\delta N` 为 `correction` 的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。支持的数据类型：mindspore.float16、mindspore.float32。
        - **dim** (Union[int, tuple(int), list(int)], 可选) - 指定维度。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None``。

    关键字参数：
        - **correction** (int, 可选) - 样本大小和样本自由度之间的差异。默认采用贝塞尔校正。默认 ``1`` 。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。如果为 ``True`` ，则保留缩小的维度，大小为1。否则移除维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(std, mean)。

    异常：
        - **ValueError** - `dim` 超出范围。