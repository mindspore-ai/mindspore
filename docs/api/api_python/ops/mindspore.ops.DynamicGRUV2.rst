mindspore.ops.DynamicGRUV2
==========================

.. py:class:: mindspore.ops.DynamicGRUV2(direction="UNIDIRECTIONAL", cell_depth=1, keep_prob=1.0, cell_clip=-1.0, num_proj=0, time_major=True, activation='tanh', gate_order='rzh', reset_after=True, is_training=True)

    为输入序列应用一个单层GRU(gated recurrent unit)。

    .. math::

        \begin{array}{ll}
            r_{t+1} = \sigma(W_{ir} x_{t+1} + b_{ir} + W_{hr} h_{(t)} + b_{hr}) \\
            z_{t+1} = \sigma(W_{iz} x_{t+1} + b_{iz} + W_{hz} h_{(t)} + b_{hz}) \\
            n_{t+1} = \tanh(W_{in} x_{t+1} + b_{in} + r_{t+1} * (W_{hn} h_{(t)}+ b_{hn})) \\
            h_{t+1} = (1 - z_{t+1}) * n_{t+1} + z_{t+1} * h_{(t)}
        \end{array}

    其中 :math:`h_{t+1}` 是在时刻t+1的隐藏状态， :math:`x_{t+1}` 是时刻t+1的输入， :math:`h_{t}` 为时刻t的隐藏状态或时刻0的初始隐藏状态。 :math:`r_{t+1}` 、 :math:`z_{t+1}` 、 :math:`n_{t+1}` 分别为重置门、更新门和当前候选集。
    :math:`W` ， :math:`b` 为可学习权重和偏置。
    :math:`\sigma` 是sigmoid激活函数， :math:`*` 为Hadamard乘积。

    参数：
        - **direction** (str) - 指定GRU方向，str类型。默认值："UNIDIRECTIONAL"。目前仅支持"UNIDIRECTIONAL"。
        - **cell_depth** (int) - GRU单元深度。默认值：1。
        - **keep_prob** (float) - Dropout保留概率。默认值：1.0。
        - **cell_clip** (float) - 输出裁剪率。默认值：-1.0。
        - **num_proj** (int) - 投影维度。默认值：0。
        - **time_major** (bool) - 如为True，则指定输入的第一维度为序列长度 `num_step` ，如为False则第一维度为 `batch_size` 。默认值：True。
        - **activation** (str) - 字符串，指定activation类型。默认值："tanh"。目前仅支持取值"tanh"。
        - **gate_order** (str) - 字符串，指定weight和bias中门的排列顺序，可选值为"rzh"或"zrh"。默认值："rzh"。"rzh"代表顺序为：重置门、更新门、隐藏门。"zrh"代表顺序为：更新门，重置门，隐藏门。
        - **reset_after** (bool) - 是否在矩阵乘法后使用重置门。默认值：True。
        - **is_training** (bool) - 是否为训练模式。默认值：True。

    输入：
        - **x** (Tensor) - 输入词序列。shape: :math:`(\text{num_step}, \text{batch_size}, \text{input_size})` 。数据类型支持float16。
        - **weight_input** (Tensor) - 权重 :math:`W_{\{ir,iz,in\}}` 。
          shape： :math:`(\text{input_size}, 3 \times \text{hidden_size})` 。
          数据类型支持float16。
        - **weight_hidden** (Tensor) - 权重 :math:`W_{\{hr,hz,hn\}}` 。
          shape： :math:`(\text{hidden_size}, 3 \times \text{hidden_size})` 。
          数据类型支持float16。
        - **bias_input** (Tensor) - 偏差 :math:`b_{\{ir,iz,in\}}` 。shape： :math:`(3 \times \text{hidden_size})` ，或 `None` 。与输入 `init_h` 的数据类型相同。
        - **bias_hidden** (Tensor) - 偏差 :math:`b_{\{hr,hz,hn\}}` 。shape： :math:`(3 \times \text{hidden_size})` ，或 `None` 。与输入 `init_h` 的数据类型相同。
        - **seq_length** (Tensor) - 每个batch中序列的长度。shape： :math:`(\text{batch_size})` 。
          目前仅支持 `None` 。
        - **init_h** (Tensor) - 初始隐藏状态。shape： :math:`(\text{batch_size}, \text{hidden_size})` 。
          数据类型支持float16和float32。

    输出：
        - **y** (Tensor) - Tensor，shape：

          - :math:`(num\_step, batch\_size, min(hidden\_size, num\_proj))` ，如果 `num_proj` 大于0,
          - :math:`(num\_step, batch\_size, hidden\_size)` ，如果 `num_proj` 等于0。
            
          与 `bias_type` 数据类型相同。

        - **output_h** (Tensor) - Tensor，shape： :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})` 。与 `bias_type` 数据类型相同。
        - **update** (Tensor) - Tensor，shape： :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})` 。与 `bias_type` 数据类型相同。
        - **reset** (Tensor) - Tensor，shape： :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})` 。与 `bias_type` 数据类型相同。
        - **new** (Tensor) - Tensor，shape： :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})` 。与 `bias_type` 数据类型相同。
        - **hidden_new** (Tensor) - Tensor，shape： :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})` 。与 `bias_type` 数据类型相同。

          关于 `bias_type` :

          - 如果 `bias_input` 和 `bias_hidden` 均为 `None` ，则 `bias_type` 为 `init_h` 的数据类型。
          - 如果 `bias_input` 不为 `None` ，则 `bias_type` 为 `bias_input` 的数据类型。
          - 如果 `bias_input` 为 `None` 而 `bias_hidden` 不为 `None` ，则 `bias_type` 为 `bias_hidden` 的数据类型。

    异常：
        - **TypeError** - `direction` 、 `activation` 或 `gate_order` 不是str。
        - **TypeError** - `cell_depth` 或 `num_proj` 不是int类型。
        - **TypeError** - `keep_prob` 或 `cell_clip` 不是float类型。
        - **TypeError** - `time_major` 、 `reset_after` 或 `is_training` 不是bool类型。
        - **TypeError** - `x` 、 `weight_input` 、 `weight_hidden` 、 `bias_input` 、 `bias_hidden` 、 `seq_length` 或 `ini_h` 不是Tensor。
        - **TypeError** - `x` 、 `weight_input` 或 `weight_hidden` 的数据类型非float16。
        - **TypeError** - `init_h` 数据类型非float16或float32。
