.. py:class:: mindspore.parallel.nn.FeedForward(hidden_size, ffn_hidden_size, dropout_rate, hidden_act="gelu", expert_num=1, param_init_type=mstype.float32, parallel_config=default_dpmp_config)

    具有两层线性层的多层感知器，并行在最终输出上使用Dropout。第一层前馈层将输入维度从hidden_size投影到ffn_hidden_size，并在中间应用激活层。第二个线性将该维度从ffn_hidden_size投影到hidden_size。配置parallel_config之后，
    第一个前馈层的权重将在输入维度上被分片，第二个线性在输出维度上进行切分。总体过程如下

    .. math:
        Dropout((xW_1+b_1)W_2 + b_2))

    其中 :math:`W_1, W_2, b_1` 和 :math:`b_2` 为可训练参数。

    **参数：**

    - **hidden_size** (int) - 表示输入的维度。
    - **ffn_hidden_size** (int) - 表示中间隐藏大小。
    - **dropout_rate** (float) - 表示第二个线性输出的丢弃率。
    - **hidden_act** (str) - 表示第一层前馈层的激活。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。默认值：gelu。
    - **expert_num** (int) - 表示线性中使用的专家数量。对于expert_num > 1用例，使用BatchMatMul。BatchMatMul中的第一个维度表示expert_num。默认值：1
    - **param_init_type** (dtype.Number) - 表示参数初始化类型。其值应为dtype.float32或dtype.float16。默认值：dtype.float32
    - **parallel_config** (OpParallelConfig) - 表示并行配置。更多详情，请参见 `OpParallelConfig` 。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    **输入：**

    - **x** (Tensor) - 应为 `[batch, seq_length, hidden_size]或[batch * seq_length, hidden_size]` 。表示浮点Tensor。

    **输出：**

    Tensor，表示映射后该层的输出。shape为 `[batch, seq_length, hidden_size]或[batch * seq_length, hidden_size]` 。

    **异常：**

    - **ValueError** - `hidden_act` 不是字符串。
    - **TypeError** - `parallel_config` 不是OpParallelConfig的子类。
    - **ValueError** - `ffn_hidden_size` 不是parallel_config中model_parallel的倍数。
    - **ValueError** - `hidden_size` 不是parallel_config中model_parallel的倍数。
