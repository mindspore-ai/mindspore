mindspore.Tensor.swapaxes
=========================

.. py:method:: mindspore.Tensor.swapaxes(axis1, axis2)

    交换Tensor的两个维度。

    参数：
        - **axis1** (int) - 第一个维度。
        - **axis2** (int) - 第二个维度。

    返回：
        转化后的Tensor，与输入具有相同的数据类型。

    异常：
        - **TypeError** - `axis1` 或 `axis2` 不是整数。
        - **ValueError** - `axis1` 或 `axis2` 不在 `[-ndim, ndim-1]` 范围内。