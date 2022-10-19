mindspore.Tensor.randint_like
==============================

.. py:method:: mindspore.Tensor.randint_like(self, high, low=0, seed=None)

    返回与输入张量大小相同的张量，数值为区间[low，high]上的随机数，如果只输入一个int类型的数据，默认值为high，如果输入两个整数，则分别为low和high。

    参数：
        - **low** (int，可选) – 要从分布中提取的最小整数。默认值：0。
        - **high** (int) – 高于要从分布中提取的最高整数的一个。
        - **seed** (int，可选) - 设置随机种子(0到2**32)。

    返回：
        Tensor，形状与self相同。

    异常：
        - **TypeError** - 如果input_sensor的数据类型不是int或float。

    平台：
        ``Ascend````GPU```` CPU``
