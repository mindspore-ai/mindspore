mindspore.mint.nn.Embedding
===========================

.. py:class:: mindspore.mint.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, dtype=None)

    以 `input` 中的值作为索引，从 `weight` 中查询对应的embedding向量。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 在Ascend后端， `input` 的值非法将导致不可预测的行为。

    参数：
        - **num_embeddings** (int) - embedding词典的大小。
        - **embedding_dim** (int) - 每个embedding向量的长度。
        - **padding_idx** (int, 可选) - 若指定该参数，则该索引处的embedding向量不会在训练中更新。对于新建的Embedding层，该索引处的embedding向量默认为全零值。取值范围： `[-num_embeddings, num_embeddings)` 。默认 ``None`` 。
        - **max_norm** (float, 可选) - 若指定该参数，则范数超过指定值的每个embedding向量将被重新归一化，使其范数等于指定值，更新公式：:math:`\frac{max\_norm}{result+1e^{-7}}` 。默认 ``None`` 。
        - **norm_type** (float, 可选) - 指定p-范数计算中的p值。默认 ``2.0`` 。
        - **scale_grad_by_freq** (bool, 可选) - 如果值为 ``True`` ，则反向梯度会按照 `input` 中索引值重复的次数进行缩放。默认 ``False`` 。
        - **sparse** (bool, 可选) - 如果值为 ``True`` ， 则 `weight` 为稀疏Tensor，该场景暂未支持。默认 ``False`` 。
        - **_weight** (Tensor, 可选) - 用于初始化Embedding的权重。如果为 ``None``，则权重从 :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` 初始化。默认 ``None`` 。
        - **_freeze** (bool, 可选) - 是否冻结该模型的可学习权重 `weight` 。默认 ``False`` 。
        - **dtype** (mindspore.dtype, 可选) - 指定Embedding权重 `weight` 的类型。当 `_weight` 不是 ``None`` 时该参数无效。默认 ``None`` 。

    可变参数：
        - **weight** (Parameter) - 模型的可学习参数，shape为 `(num_embeddings, embedding_dim)`，从 :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` 或 `_weight` 初始化。

    输入：
        - **input** (Tensor) - 索引tensor，取值范围： `[0, num_embeddings)` 。

    输出：
        Tensor 。
