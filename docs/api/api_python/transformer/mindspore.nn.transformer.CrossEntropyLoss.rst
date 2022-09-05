.. py:class:: mindspore.nn.transformer.CrossEntropyLoss(parallel_config=default_dpmp_config)

    计算输入和输出之间的交叉熵损失。

    参数：
        - **parallel_config** (OpParallelConfig) - 表示并行配置。默认值为 `default_dpmp_config` ，表示一个带有默认参数的 `OpParallelConfig` 实例。

    输入：
        - **logits** (Tensor) - shape为(N, C)的Tensor。表示的输出logits。其中N表示任意大小的维度，C表示类别个数。数据类型必须为float16或float32。
        - **labels** (Tensor) - shape为(N, )的Tensor。表示样本的真实标签，其中每个元素的取值区间为[0,C)。
        - **input_mask** (Tensor) - shape为(N, )的Tensor。input_mask表示是否有填充输入。1表示有效，0表示无效，其中元素值为0的位置不会计算进损失值。

    输出：
        Tensor，表示对应的交叉熵损失。
