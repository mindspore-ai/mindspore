mindspore.nn.RNN
=================

.. py:class:: mindspore.nn.RNN(*args, **kwargs)

    循环神经网络（RNN）层，其使用的激活函数为tanh或relu。

    将具有 :math:`\tanh` 或 :math:`\text{ReLU}` 非线性的RNN层应用到输入。

    对输入序列中的每个元素，每层的计算公式如下：

    .. math::
        h_t = activation(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    这里的 :math:`h_t` 是在 `t` 时刻的隐藏状态， :math:`x_t` 是在 `t` 时刻的输入， :math:`h_{(t-1)}` 是上一层在 :math:`t-1` 时刻的隐藏状态，或初始隐藏状态。如果 `nonlinearity` 是'relu'，则使用 :math:`\text{ReLU}` 而不是 :math:`\tanh` 。

    参数：
        - **input_size** (int) - 输入层输入的特征向量维度。
        - **hidden_size** (int) - 隐藏层输出的特征向量维度。
        - **num_layers** (int) - 堆叠RNN的层数。默认值：1。
        - **nonlinearity** (str) - 用于选择非线性激活函数。取值可为'tanh'或'relu'。默认值：'tanh'。
        - **has_bias** (bool) - Cell是否有偏置项 `b_ih` 和 `b_hh` 。默认值：True。
        - **batch_first** (bool) - 指定输入 `x` 的第一个维度是否为batch_size。默认值：False。
        - **dropout** (float) - 指的是除第一层外每层输入时的Dropout概率。Dropout的范围为[0.0, 1.0)。默认值：0.0。
        - **bidirectional** (bool) - 指定是否为双向RNN，如果bidirectional=True，则num_directions=2，否则为1。默认值：False。

    输入：
        - **x** (Tensor) - 数据类型为mindspore.float32或mindspore.float16，shape为 :math:`(seq\_len, batch\_size, input\_size)` 或 :math:`(batch\_size, seq\_len, input\_size)` 的Tensor。
        - **hx** (Tensor) - 数据类型为mindspore.float32或mindspore.float16，shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。 `hx` 的数据类型与 `x` 相同。
        - **seq_length** (Tensor) - 输入batch的序列长度，Tensor的shape为 :math:`(batch\_size)` 。此输入指明真实的序列长度，以避免使用填充后的元素计算隐藏状态，影响最后的输出。当 `x` 被填充元素时，建议使用此输入。默认值：None。

    输出：
        Tuple，包含(`output`, `hx_n`)的tuple。

        - **output** (Tensor) - shape为 :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` 或 :math:`(batch\_size, seq\_len, num\_directions * hidden\_size)` 的Tensor。
        - **hx_n** (Tensor) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。

    异常：
        - **TypeError** - `input_size` ， `hidden_size` 或 `num_layers` 不是int。
        - **TypeError** - `has_bias` ， `batch_first` 或 `bidirectional` 不是bool。
        - **TypeError** - `dropout` 不是float。
        - **ValueError** - `dropout` 不在[0.0, 1.0)范围内。
        - **ValueError** - `nonlinearity` 不在['tanh', 'relu']中。