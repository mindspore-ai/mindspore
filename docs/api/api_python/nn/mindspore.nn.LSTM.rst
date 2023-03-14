mindspore.nn.LSTM
==================

.. py:class:: mindspore.nn.LSTM(*args, **kwargs)

    长短期记忆（LSTM）网络，根据输入序列和给定的初始状态计算输出序列和最终状态。

    在LSTM模型中，有两条管道连接两个连续的Cell，一条是Cell状态管道，另一条是隐藏状态管道。将两个连续的时间节点表示为 :math:`t-1` 和 :math:`t` 。指定在 :math:`t` 时刻输入 :math:`x_t` ，在 :math:`{t-1}` 时刻的隐藏状态 :math:`h_{t-1}` 和Cell状态 :math:`c_{t-1}` 。在 :math:`t` 时刻的Cell状态和隐藏状态使用门控机制计算得到。输入门 :math:`i_t` 计算出候选值。遗忘门 :math:`f_t` 决定是否让上一时刻学到的信息通过或部分通过。输出门 :math:`o_t` 决定哪些信息输出。候选Cell状态 :math:`\tilde{c}_t` 是用当前输入计算的。最后，使用遗忘门、输入门、输出门计算得到当前时刻的Cell状态 :math:`c_{t}` 和隐藏状态 :math:`h_{t}` 。完整的公式如下。

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    其中 :math:`\sigma` 是sigmoid激活函数， :math:`*` 是乘积。 :math:`W, b` 是公式中输出和输入之间的可学习权重。例如， :math:`W_{ix}, b_{ix}` 是用于从输入 :math:`x` 转换为 :math:`i` 的权重和偏置。

    详细信息可见论文 `LONG SHORT-TERM MEMORY <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ 和 `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_ 。

    LSTM隐藏了整个循环神经网络在序列时间步(Time step)上的循环，送入输入序列、初始状态，即可获得每个时间步的隐藏状态(hidden state)拼接而成的矩阵，以及最后一个时间步对应的隐状态。我们使用最后的一个时间步的隐藏状态作为输入句子的编码特征，送入下一层。公式为：

    .. math::
        h_{0:n},(h_{n}, c_{n}) = LSTM(x_{0:n},(h_{0},c_{0}))

    参数：
        - **input_size** (int) - 输入的大小。
        - **hidden_size** (int) - 隐藏状态大小。
        - **num_layers** (int) - 网络层数。默认值：1。
        - **has_bias** (bool) - Cell是否有偏置项 `b_{ih}` 和 `b_{fh}`。默认值：True。
        - **batch_first** (bool) - 指定输入 `x` 的第一个维度是否为batch_size。默认值：False。
        - **dropout** (float, int) - 指的是除第一层外每层输入时的dropout概率。默认值：0。dropout的范围为[0.0, 1.0)。
        - **bidirectional** (bool) - 是否为双向LSTM。默认值：False。

    输入：
        - **x** (Tensor) - shape为 (seq_len, batch_size, `input_size`)或(batch_size, seq_len, `input_size`)的Tensor。
        - **hx** (tuple) - 两个Tensor(h_0,c_0)的元组，数据类型为mindspore.float32或mindspore.float16，shape为(num_directions * `num_layers`, batch_size, `hidden_size`)。`hx` 的数据类型必须与 `x` 相同。
        - **seq_length** (Tensor) - 输入batch的序列长度。Tensor的shape 为 :math:`(batch\_size)` 。默认：None。这里输入指明真实的序列长度，以避免使用填充后的元素计算隐藏状态，影响最后的输出。推荐这种输入方法。

    输出：
        Tuple，包含 (`output`, (`h_n`, `c_n`))的元组。

        - **output** (Tensor) - 形状为(seq_len, batch_size, num_directions * `hidden_size`)的Tensor。
        - **hx_n** (tuple) - 两个Tensor (h_n, c_n)的元组，shape都是(num_directions * `num_layers`, batch_size, `hidden_size`)。

    异常：
        - **TypeError** - `input_size`， `hidden_size` 或  `num_layers` 不是int。
        - **TypeError** - `has_bias` ， `batch_first` 或 `bidirectional` 不是bool。
        - **TypeError** - `dropout` 既不是float也不是int。
        - **ValueError** - `dropout` 不在[0.0, 1.0)范围内。
