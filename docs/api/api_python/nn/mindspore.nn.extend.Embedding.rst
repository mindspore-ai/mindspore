mindspore.nn.extend.Embedding
=============================

.. py:class:: mindspore.nn.extend.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, _weight=None, dtype=mstype.float32)

    嵌入层。

    用于存储词向量并使用索引进行检索。层的输入是一个索引列表的Tensor，从权重中查询对应位置的embedding向量。

    .. warning::
        在Ascend后端，`input` 的值非法将导致不可预测的行为。

    参数：
        - **num_embeddings** (int) - 词典的大小。
        - **embedding_dim** (int) - 每个嵌入向量的大小。
        - **padding_idx** (int, 可选) - 如果给定非 ``None`` 值， `padding_idx` 对应索引的权重在训练过程中不会被更新。初始化时， `padding_idx` 对应索引的权重将被初始化为0。有效值范围： `[-num_embeddings, num_embeddings)` 。默认值： ``None`` 。
        - **max_norm** (float, 可选) - 如果给定非 ``None`` 值，则先求出 `input` 指定位置的 `weight` 的p-范数结果reslut（p的值通过 `norm_type` 指定），然后对 `result > max_norm` 位置的权重进行更新，更新公式：:math:`\frac{max\_norm}{result+1e^{-7}}`。默认值 ``None`` 。
        - **norm_type** (float, 可选) - 指定p-范数计算中的p值。默认值 ``2.0`` 。
        - **scale_grad_by_freq** (bool, 可选) - 如果值为 ``True`` ，则反向梯度值会按照 `input` 中索引值重复的次数进行缩放。默认值 ``False`` 。
        - **_weight** (Tensor, 可选) - 用于初始化Embedding层的权重初始化，如果为 ``None`` ，权重将使用正态分布 :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` 进行初始化。默认值： ``None`` 。
        - **dtype** (mindspore.dtype, 可选) - 权重初始化的数据类型。如果 `_weight` 给定非 ``None`` 值，该参数将失效。默认值： ``mindspore.float32`` 。

    输入：
        - **input** (Tensor) - 用于查找embedding向量的索引。该数据类型可以是int32或int64，取值范围： `[0, num_embeddings)` 。

    输出：
        Tensor的shape :math:`(*input.shape, embedding_dim)` ，类型与权重一致。

    异常：
        - **TypeError** - `num_embeddings` 的类型不是int。
        - **TypeError** - `embedding_dim` 的类型不是int。
        - **ValueError** - `padding_idx` 取值不在有效范围。
        - **TypeError** - `max_norm` 的类型不是float。
        - **TypeError** - `norm_type` 的类型不是float。
        - **TypeError** - `scale_grad_by_freq` 的类型不是bool。
        - **TypeError** - `dtype` 的类型不是mindspore.dtype。
