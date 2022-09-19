mindspore.Tensor.trace
======================

.. py:method:: mindspore.Tensor.trace(offset=0, axis1=0, axis2=1, dtype=None)

    在Tensor的对角线方向上的总和。

    参数：
        - **offset** (int, 可选) - 对角线与主对角线的偏移。可以是正值或负值。默认为主对角线。
        - **axis1** (int, 可选) - 二维子数组的第一轴，对角线应该从这里开始。默认为第一轴(0)。
        - **axis2** (int, 可选) - 二维子数组的第二轴，对角线应该从这里开始。默认为第二轴。
        - **dtype** (`mindspore.dtype` , 可选) - 默认值为None。覆盖输出Tensor的dtype。

    返回：
        Tensor，对角线方向上的总和。

    异常：
        **ValueError** - 输入Tensor的维度少于2。