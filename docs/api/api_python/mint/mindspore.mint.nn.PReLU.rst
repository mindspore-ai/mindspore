mindspore.mint.nn.PReLU
=======================

.. py:class:: mindspore.mint.nn.PReLU(num_parameters=1, init=0.25, dtype=None)

    逐元素计算PReLU（PReLU Activation Operator）激活函数。

    公式定义为：

    .. math::

        PReLU(x_i)= \max(0, x_i) + w * \min(0, x_i),

    其中 :math:`x_i` 是输入tensor。

    这里 :math:`w` 是一个可学习的参数，默认初始值0.25。
    当带参数调用时每个通道上学习一个 :math:`w` 。如果不带参数调用时，则将在所有通道中共享单个参数 :math:`w` 。

    PReLU函数图：

    .. image:: ../images/PReLU2.png
        :align: center

    .. note::
        - 通道数是输入的第二个维度值。当输入的维度小于2时，则没有通道维度，此时通道数等于1。
        - 在GE模式下，输入tensor的秩必须大于1，否则会触发报错。

    参数：
        - **num_parameters** (int，可选) - 参数 :math:`w` 的数量。合法值为：1或输入tensor `input` 的通道数。默认 `1` 。
        - **init** (float，可选) - 参数的初始值。默认值： ``0.25`` 。
        - **dtype** (mindspore.dtype，可选) - 参数的dtype。默认值： ``None`` 。支持的数据类型是{float16, float32, bfloat16}。

    输入：
        - **input** (Tensor) - 输入tensor。

    输出：
        Tensor。
