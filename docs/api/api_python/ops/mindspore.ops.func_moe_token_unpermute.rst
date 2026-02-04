mindspore.ops.moe_token_unpermute
======================================

.. py:function:: mindspore.ops.moe_token_unpermute(permuted_tokens, sorted_indices, probs=None, padded_mode=False, restore_shape=None)

    根据排序的索引对已排列的标记进行反排列，并可选择将标记与其对应的概率合并。

    .. warning::
        - 仅支持 Atlas A2 训练系列产品。
        - `sorted_indices` 必须不存在重复值，否则结果未定义。

    参数：
        - **permuted_tokens** (Tensor) - 要进行反排列的已排列标记的Tensor。
          shape为 :math:`[num\_tokens * topk, hidden\_size]`，其中 `num_tokens`、 `topk` 和 `hidden_size` 都是正整数。
        - **sorted_indices** (Tensor) - 用于反排列标记的排列索引Tensor。shape为 :math:`[num\_tokens * topk,]` ，
          其中 `num_tokens` 和 `topk` 都是正整数。仅支持int32的数据类型。
        - **probs** (Tensor，可选) - 与已排列标记对应的概率Tensor。如果提供，反排列的标记将与其相应的概率合并。
          shape为 :math:`[num\_tokens, topk]` ，其中 `num_tokens` 和 `topk` 都是正整数。默认 ``None`` 。
        - **padded_mode** (bool, 可选) - 如果为 ``True``，表示索引被填充，以表示每个专家选择的标记。默认 ``False``。
        - **restore_shape** (Union[tuple[int], list[int]]，可选) - 排列之前的输入形状，仅在填充模式下使用。默认 ``None``。

    返回：
        Tensor，类型与 `permuted_tokens` 一致。如果 `padded_mode` 为 ``False``， 则shape为[`num_tokens`, `hidden_size`]。
        如果 `padded_mode` 为 ``True``，则shape由 `restore_shape` 指定。

    异常：
        - **TypeError** - `permuted_tokens` 不是Tensor。
        - **ValueError** - 仅支持 `padded_mode` 为 ``False``。
