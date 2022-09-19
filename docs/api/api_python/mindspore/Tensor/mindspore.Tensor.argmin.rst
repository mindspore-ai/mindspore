mindspore.Tensor.argmin
=======================

.. py:method:: mindspore.Tensor.argmin(axis=None)

    返回指定轴上最小值的索引。

    参数：
        - **axis** (int, 可选) - 返回扁平化Tensor的最小值序号，否则返回指定轴方向上的最小值序号。默认值: None。

    返回：
        Tensor，最小Tensor的索引。它与原始Tensor具有相同的shape，但移除了轴方向上的维度。

    异常：
        - **ValueError** - 入参axis的设定值超出了范围。