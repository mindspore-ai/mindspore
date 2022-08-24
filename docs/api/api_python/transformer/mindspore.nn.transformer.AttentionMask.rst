.. py:class:: mindspore.nn.transformer.AttentionMask(seq_length, parallel_config=default_dpmp_config)

    从输入掩码中获取下三角矩阵。输入掩码是值为1或0的二维Tensor (batch_size, seq_length)，1表示当前位置是一个有效的标记，0则表示当前位置不是一个有效的标记。

    参数：
        - **seq_length** (int) - 表示输入Tensor的序列长度。
        - **parallel_config** (OpParallelConfig) - 表示并行配置。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    输入：
        - **input_mask** (Tensor) - 掩码矩阵，shape为(batch_size, seq_length)，表示每个位置是否为有效输入。

    输出：
        Tensor，表示shape为(batch_size, seq_length, seq_length)的注意力掩码矩阵。

    异常：
        - **TypeError** - `seq_length` 不是整数。
        - **ValueError** - `seq_length` 不是正数。
        - **TypeError** - `parallel_config` 不是OpParallelConfig的子类。
