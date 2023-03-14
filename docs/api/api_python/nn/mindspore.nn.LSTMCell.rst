mindspore.nn.LSTMCell
======================

.. py:class:: mindspore.nn.LSTMCell(*args, **kwargs)

    长短期记忆网络单元（LSTMCell）。

    公式如下：

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    其中 :math:`\sigma` 是sigmoid函数， :math:`*` 是乘积。 :math:`W, b` 是公式中输出和输入之间的可学习权重。例如，:math:`W_{ix}, b_{ix}` 是用于从输入 :math:`x` 转换为 :math:`i` 的权重和偏置。详见论文 `LONG SHORT-TERM MEMORY <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ 和 `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_ 。

    nn层封装的LSTMCell可以简化为如下公式：

    .. math::
        h^{'},c^{'} = LSTMCell(x, (h_0, c_0))

    参数：
        - **input_size** (int) - 输入的大小。
        - **hidden_size** (int) - 隐藏状态大小。
        - **has_bias** (bool) - cell是否有偏置 `b_ih` 和 `b_hh` 。默认值：True。

    输入：
        - **x** (Tensor) - shape为(batch_size, `input_size`)的Tensor。
        - **hx** (tuple) - 两个Tensor(h_0,c_0)的元组，其数据类型为mindspore.float32，shape为(batch_size, `hidden_size`)。 `hx` 的数据类型必须与 `x` 相同。

    输出：
        - **hx'** (Tensor) - 两个Tensor(h', c')的元组，其shape为(batch_size, `hidden_size`)。

    异常：
        - **TypeError** - `input_size`， `hidden_size` 不是整数。
        - **TypeError** - `has_bias` 不是bool。