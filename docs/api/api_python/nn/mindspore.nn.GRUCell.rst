mindspore.nn.GRUCell
=====================

.. py:class:: mindspore.nn.GRUCell(input_size: int, hidden_size: int, has_bias: bool = True)

    GRU（Gate Recurrent Unit）称为门控循环单元。

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    这里 :math:`\sigma` 是sigmoid激活函数， :math:`*` 是乘积。 :math:`W, b` 是公式中输出和输入之间的可学习权重。例如， :math:`W_{ir}, b_{ir}` 是用于将输入 :math:`x` 转换为 :math:`r` 的权重和偏置。详见论文 `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation <https://aclanthology.org/D14-1179.pdf>`_ 。

    参数：
        - **input_size** (int) - 输入的大小。
        - **hidden_size** (int) - 隐藏状态大小。
        - **has_bias** (bool) - cell是否有偏置项 `b_in` 和 `b_hn` 。默认值：True。

    输入：
        - **x** (Tensor) - shape为(batch_size, `input_size` )的Tensor。
        - **hx** (Tensor) - 数据类型为mindspore.float32、shape为(batch_size, `hidden_size` )的Tensor。 `hx` 的数据类型必须与 `x` 相同。

    输出：
        - **hx'** (Tensor) - shape为(batch_size, `hidden_size`)的Tensor。

    异常：
        - **TypeError** - `input_size` 、 `hidden_size` 不是int。
        - **TypeError** - `has_bias` 不是bool值。
