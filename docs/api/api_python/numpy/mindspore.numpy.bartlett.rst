mindspore.numpy.bartlett
=================================

.. py:function:: mindspore.numpy.bartlett(M)

    用于生成Bartlett窗口。Bartlett窗口是一种特殊的三角形窗口函数，它在信号处理和频谱分析中经常使用。Bartlett窗口在时域上是对称的，幅度逐渐减小。

    参数：
        - **M** (int) - 输出窗口的长度。如果为零或小于零，则返回一个空数组。

    返回：
        Tensor，长度为 ``M`` 的Bartlett窗口，其中它的第一个值和最后一个值为 :math:`0` ，中间部分为尖峰，即三角形形状（仅当样本数为奇数时才会出现值 :math:`1` ）。

    异常：
        - **TypeError** - 如果输入 ``M`` 不是一个int型数据。