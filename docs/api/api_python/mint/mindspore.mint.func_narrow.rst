mindspore.mint.narrow
=====================

.. py:function:: mindspore.mint.narrow(input, dim, start, length)

    沿着指定的轴，指定起始位置获取指定长度的Tensor。

    参数：
        - **input** (Tensor) - 需要计算的Tensor。
        - **dim** (int) - 指定的轴。
        - **start** (int) - 指定起始位置。
        - **length** (int) - 指定长度。

    返回：
        Tensor。

    异常：
        - **TypeError** - 如果输入不是Tensor、元组或Tensor组成的列表。
