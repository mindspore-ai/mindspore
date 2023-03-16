mindspore.ops.DynamicRNN
========================

.. py:class:: mindspore.ops.DynamicRNN(cell_type='LSTM', direction='UNIDIRECTIONAL', cell_depth=1, use_peephole=False, keep_prob=1.0, cell_clip=-1.0, num_proj=0, time_major=True, activation='tanh', forget_bias=0.0, is_training=True)

    将循环神经网络应用到输入上。当前仅支持LSTM。

    .. math::
        \begin{array}{ll} \\
            i_{t+1} = \sigma(W_{ix} x_{t+1} + b_{ix} + W_{ih} h_{(t)} + b_{ih}) \\
            f_{t+1} = \sigma(W_{fx} x_{t+1} + b_{fx} + W_{fh} h_{(t)} + b_{fh}) \\
            \tilde{c}_{t+1} = \tanh(W_{cx} x_{t+1} + b_{cx} + W_{ch} h_{(t)} + b_{ch}) \\
            o_{t+1} = \sigma(W_{ox} x_{t+1} + b_{ox} + W_{oh} h_{(t)} + b_{oh}) \\
            c_{t+1} = f_{t+1} * c_{(t)} + i_t * \tilde{c}_{t+1} \\
            h_{t+1} = o_{t+1} * \tanh(c_{t+1}) \\
        \end{array}

    其中，
    :math:`h_{t+1}` 是在 `t+1` 时刻的隐藏状态。 
    :math:`x_{t+1}` 是在 `t+1` 时刻的输入。
    :math:`h_{t}` 是在 `t` 时刻的隐藏状态或在 `0` 时刻的初始隐藏状态。
    :math:`\sigma` 是sigmoid函数， :math:`*` 是 `Hadamard` 积。
    :math:`W, b` 是公式中输出和输入之间的可学习权重。 

    例如， :math:`W_{ix}, b_{ix}` 是把 :math:`x` 转换为 :math:`i` 的权重和偏置。

    参数：
        - **cell_type** (str) - 指定Cell类型。当前仅支持LSTM。默认值：LSTM。
        - **direction** (str) - 指定单向或双向。默认值：UNIDIRECTIONAL。当前仅支持UNIDIRECTIONAL。
        - **cell_depth** (int) - 指定cell的层数。默认值：1。
        - **use_peephole** (bool) - 是否使用"peephole connections"。默认值：False。
        - **keep_prob** (float) - 指定保留率，即每个元素被保留的概率。1.0表示所有元素全部保留。默认值：1.0。
        - **cell_clip** (float) - 将Cell裁剪到指定的值，负值表示禁用。默认值：-1.0。
        - **num_proj** (int) - 投影矩阵的输出维数。默认值：0。
        - **time_major** (bool) - 指定输入 `x` 的数据排列格式。如果为True，格式为 :math:`(num\_step, batch\_size, input\_size)`，如果为False，格式为：:math:`(batch\_size, num\_step, input\_size)` 。默认值：True。当前仅支持True。
        - **activation** (str) - 指定激活函数。默认值：tanh。当前仅支持tanh。
        - **forget_bias** (float) - 指定遗忘门的偏置。默认值：0.0。
        - **is_training** (bool) - 指定是否开启训练。默认值：True。

    输入：
        - **x** (Tensor) - 输入的词汇。shape为 :math:`(num\_step, batch\_size, input\_size)` 的Tensor。数据类型必须为float16。
        - **w** (Tensor) - 输入的权重。shape为 :math:`(input\_size + hidden\_size, 4 * hidden\_size)` 的Tensor。数据类型必须为float16。
        - **b** (Tensor) - 输入的偏置。shape为 :math:`(4 * hidden\_size)` 的Tensor。数据类型必须为float16或float32。
        - **seq_length** (Tensor) - 每个批次中句子的真实长度。shape为 :math:`(batch\_size, )` 的Tensor。当前仅支持None。
        - **init_h** (Tensor) - 在初始时刻的隐藏状态。shape为 :math:`(1, batch\_size, hidden\_size)` 的Tensor。数据类型必须为float16。
        - **init_c** (Tensor) - 在初始时刻的Cell状态。shape为 :math:`(1, batch\_size, hidden\_size)` 的Tensor。数据类型必须为float16。

    输出：
        - **y** (Tensor) - 所有时刻输出层的输出向量，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。
        - **output_h** (Tensor) - 所有时刻输出层的输出向量，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型为float16。
        - **output_c** (Tensor) - 所有时刻的Cell状态的输出向量，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。
        - **i** (Tensor) - 更新输入门的权重，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。
        - **j** (Tensor) - 更新新门的权重，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。
        - **f** (Tensor) - 更新遗忘门的权重，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型输入 `b` 相同。
        - **o** (Tensor) - 更新输出门的权重，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。
        - **tanhct** (Tensor) - 更新tanh的权重，shape为 :math:`(num\_step, batch\_size, hidden\_size)` 的Tensor。数据类型与输入 `b` 相同。

    异常：
        - **TypeError** - `cell_type` 、 `direction` 或 `activation` 不是str。
        - **TypeError** - `cell_Deep` 或 `num_proj` 不是int。
        - **TypeError** - `keep_prob` 、 `cell_clip` 或 `forget_bias` 不是float。
        - **TypeError** - `use_peehpol` 、 `time_major` 或 `is_training` 不是bool。
        - **TypeError** - `x` 、 `w` 、 `b` 、 `seq_length` 、 `init_h` 或 `init_c` 不是Tensor。
        - **TypeError** - `x` 、 `w` 、 `init_h` 或 `nit_c` 的数据类型不是float16。
        - **TypeError** - `b` 的数据类型既不是float16也不是float32。
