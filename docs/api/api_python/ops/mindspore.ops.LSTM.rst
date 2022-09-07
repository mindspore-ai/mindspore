mindspore.ops.LSTM
===================

.. py:class:: mindspore.ops.LSTM(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)

    长短期记忆（LSTM）网络。

    有关详细信息，请参见 :class:`mindspore.nn.LSTM` 。

    参数：
        - **input_size** (int) - 输入的大小。
        - **hidden_size** (int) - 隐藏状态大小。
        - **num_layers** (int) - LSTM的网络层数。
        - **has_bias** (bool) - Cell是否有偏置 `b_ih` 和 `b_hh` 。
        - **bidirectional** (bool) - 是否为双向LSTM。
        - **dropout** (float) - 指的是除第一层外每层输入时的dropout概率。默认值：0。dropout的范围为[0.0, 1.0]。

    输入：
        - **input** (Tensor) - shape为 :math:`(seq\_len, batch\_size, input\_size)` 或 :math:`(batch\_size, seq\_len, input\_size)` 的Tensor。
        - **h** (tuple) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。
        - **c** (tuple) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。
        - **w** (Tensor) - 权重。

    输出：
        tuple，tuple包含( `output` , `h\_n` , `c\_n` , `reserve` , `state` )。

        - **output** (Tensor) - shape为 :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` 的Tensor。
        - **h_n** (Tensor) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。
        - **c_n** (Tensor) - shape为 :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` 的Tensor。
        - **reserve** (Tensor) - shape为 :math:`(r,1)` 。
        - **state** (Tensor) - 使用随机数生成状态及其shape为 :math:`(s,1)` 。

    异常：
        - **TypeError** - 如果 `input_size` ， `hidden_size` 或 `num_layers` 不是int。
        - **TypeError** - 如果 `has_bias` 或 `bidirectional` 不是bool。
        - **TypeError** - 如果 `dropout` 不是float。
        - **ValueError** - 如果 `dropout` 不在范围[0.0, 1.0]内。
