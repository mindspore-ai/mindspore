mindspore.nn.GRU
=================

.. py:class:: mindspore.nn.GRU(*args, **kwargs)

    GRU（Gate Recurrent Unit）称为门控循环单元网络，是循环神经网络（Recurrent Neural Network, RNN）的一种。根据输出序列和给定的初始状态计算输出序列和最终状态。

    应用GRU层到输入中。

    GRU网络模型中有两个门。一个是更新门，另一个是重置门。将两个连续的时间节点表示为 :math:`t-1` 和 :math:`t`。给定一个在时刻 :math:`t` 的输入 :math:`x_t` ，一个隐藏状态 :math:`h_{t-1}` ，在时刻 :math:`t` 的更新门和重置门使用门控机制计算。更新门 :math:`z_t` 用于控制前一时刻的状态信息被带入到当前状态中的程度。重置门 :math:`r_t` 控制前一状态有多少信息被写入到当前候选集 :math:`n_t` 上。完整的公式如下。

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    其中 :math:`\sigma` 是sigmoid激活函数， :math:`*` 是乘积。 :math:`W,b` 是公式中输出和输入之间的可学习权重。例如， :math:`W_{ir}, b_{ir}` 是用于将输入 :math:`x` 转换为 :math:`r` 的权重和偏置。详见论文 `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation <https://aclanthology.org/D14-1179.pdf>`_ 。

    .. note:: 当GRU运行在Ascend上时，hidden size仅支持16的倍数。

    参数：
        - **input_size** (int) - 输入的大小。
        - **hidden_size** (int) - 隐藏状态大小。
        - **num_layers** (int) - 网络层数。默认值：1。
        - **has_bias** (bool) - cell是否有偏置项 `b_{in}` 和 `b_{hn}` 。默认值：True。
        - **batch_first** (bool) - 指定输入 `x` 的第一个维度是否为batch_size。默认值：False。
        - **dropout** (float) - 指的是除第一层外每层输入时的Dropout概率。默认值：0.0。Dropout的范围为[0.0, 1.0)。
        - **bidirectional** (bool) - 是否为双向GRU。如果bidirectional=True，则num_directions=2，为双向GRU。否则为1，单向GRU。默认值：False。

    输入：
        - **x** (Tensor) - 数据类型为mindspore.float32、shape为(seq_len, batch_size, `input_size`)或(batch_size, seq_len, `input_size`)的tensor。     
        - **hx** (Tensor) - 数据类型为mindspore.float32、shape为(num_directions * `num_layers` , batch_size, `hidden_size` )的tensor。 `hx` 的数据类型必须与 `x` 相同。
        - **seq_length** (Tensor) - 输入batch中每个序列的长度。shape为 `(batch_size)` 的Tensor。默认值：None。此输入指示填充前的真实序列长度，避免填充元素被用于计算隐藏状态而影响最终输出。当 `x` 含填充元素时，建议使用此输入。

    输出：
        Tuple，包含(`output`, `h_n`)的tuple。

        - **output** (Tensor) - shape为(seq_len, batch_size, num_directions * `hidden_size`)或(batch_size, seq_len, num_directions * `hidden_size`)的Tensor。
        - **hx_n** (Tensor) - shape为(num_directions * `num_layers`, batch_size, `hidden_size`)的Tensor。

    异常：
        - **TypeError** - `input_size` ， `hidden_size` 或 `num_layers` 不是整数。
        - **TypeError** - `has_bias` ， `batch_first` 或 `bibound` 不是bool。
        - **TypeError** - `dropout` 既不是浮点数也不是整数。
        - **ValueError** - `dropout` 不在[0.0, 1.0)范围内。

