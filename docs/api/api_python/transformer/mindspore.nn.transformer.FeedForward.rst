.. py:class:: mindspore.nn.transformer.FeedForward(hidden_size, ffn_hidden_size, dropout_rate, hidden_act="gelu", expert_num=1, expert_group_size=None, param_init_type=mstype.float32, parallel_config=default_dpmp_config)

    具有两层线性层的多层感知器，并在最终输出上使用Dropout。第一个线性层将输入维度从hidden_size投影到ffn_hidden_size，并在中间应用激活层。第二个线性层将该维度从ffn_hidden_size投影到hidden_size。配置parallel_config之后，
    第一个线性层的权重将在输入维度上被分片，第二个线性层在输出维度上进行切分。总体过程如下：

    .. math::
        Dropout((xW_1+b_1)W_2 + b_2)

    其中 :math:`W_1, W_2, b_1` 和 :math:`b_2` 为可训练参数。

    参数：
        - **hidden_size** (int) - 表示输入的维度。
        - **ffn_hidden_size** (int) - 表示中间隐藏大小。
        - **dropout_rate** (float) - 表示第二个线性层输出的丢弃率。
        - **hidden_act** (str, nn.Cell) - 表示前馈层的激活行为。其值可为'relu'、'relu6'、'tanh'、'gelu'、'fast_gelu'、'elu'、'sigmoid'、'prelu'、'leakyrelu'、'hswish'、'hsigmoid'、'logsigmoid'等等。用户可以传入自定义的激活函数。如果用户要想在并行模式下运行此网络，自定义的激活函数必须提供 `activation_shard` 类方法。请查看类 `mindspore.nn.transformer.FeedForward` 的示例。默认值：gelu。
        - **expert_num** (int) - 表示线性层中使用的专家数量。对于expert_num > 1用例，使用BatchMatMul。BatchMatMul中的第一个维度表示expert_num。默认值：1。
        - **expert_group_size** (int) - 表示每个数据并行组收到的词语（token）数量。默认值：None。该参数只在自动并行且非策略传播模式下起作用。
        - **param_init_type** (dtype.Number) - 表示参数初始化类型。其值应为mstype.float32或mstype.float16。默认值：mstype.float32。
        - **parallel_config** (OpParallelConfig, MoEParallelConfig) - 表示配置该网络的并行度的并行配置。更多详情，请参见 :class:`mindspore.nn.transformer.OpParallelConfig` 。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    输入：
        - **x** (Tensor) - 应为 `[batch, seq_length, hidden_size]或[batch * seq_length, hidden_size]` 。表示浮点Tensor。

    输出：
        Tensor，表示映射后该层的输出。shape为 `[batch, seq_length, hidden_size]` 或 `[batch * seq_length, hidden_size]` 。

    异常：
        - **TypeError** - `hidden_act` 不是字符串或者nn.Cell。
        - **TypeError** - `parallel_config` 不是OpParallelConfig的子类。
        - **ValueError** - `ffn_hidden_size` 不是parallel_config中model_parallel的倍数。
        - **ValueError** - `hidden_size` 不是parallel_config中model_parallel的倍数。
