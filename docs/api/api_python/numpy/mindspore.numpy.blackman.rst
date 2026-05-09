mindspore.numpy.blackman
=================================

.. py:function:: mindspore.numpy.blackman(M)

    用于生成Blackman窗口。Blackman窗口使用前三项的余弦和计算得到，其设计是为了尽可能减少泄露。它接近于最优，仅次于Kaiser接口。

    参数：
        - **M** (int) - 输出窗口的长度。如果为零或小于零，则返回一个空数组。

    返回：
        Tensor，生成一个长度为 `M` 且最大值归一化为 :math:`1` 的Blackman窗口（仅当样本数为奇数时才会出现值 :math:`1` ）。

    异常：
        - **TypeError** - 如果输入 `M` 不是一个int型数据。