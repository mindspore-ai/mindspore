mindspore.numpy.hanning
=================================

.. py:function:: mindspore.numpy.hanning(M)

    返回一个Hanning窗口。Hanning窗口是一种基于加权余弦的锥度函数。

    参数：
        - **M** (int) - 输出窗口的长度。如果为零或小于零，则返回一个空数组。

    返回：
        Tensor，生成一个长度为 ``M`` 且最大值归一化为1的Hanning窗口（仅当样本数为奇数时才会出现值1）。

    异常：
        - **TypeError** - 如果 ``M`` 不是int类型。