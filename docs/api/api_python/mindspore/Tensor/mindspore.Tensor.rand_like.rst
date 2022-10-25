mindspore.Tensor.rand_like
==========================

.. py:method:: mindspore.Tensor.rand_like(self, seed=None)

    返回与填充的输入大小相同的张量，数值为区间[0,1)上均匀分布的随机数

    参数：
        seed(int，option)：设置随机种子(0到2**32)。

    返回：
        Tensor，形状与self相同。

    异常：
        - **TypeError** - 如果input_sensor的数据类型不是int或float

    平台：
        ``Ascend````GPU```` CPU``
