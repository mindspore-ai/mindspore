mindspore.ops.log_uniform_candidate_sampler
===========================================

.. py:function:: mindspore.ops.log_uniform_candidate_sampler(true_classes, num_true=1, num_sampled=5, unique=True, range_max=5, seed=0)

    使用log-uniform(Zipfian)分布对一组类别进行采样。

    .. warning::
        - 该接口已废弃，2.9.0（不含）之后版本将删除。
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    该操作从整数范围[0, `range_max` )中随机采样一个采样类( `sampled_candidates` )的Tensor。

    参数：
        - **true_classes** (Tensor) - 目标类，其数据类型为int64，shape为 :math:`(batch\_size, num\_true)` 。
        - **num_true** (int，可选) - 每个训练样本的目标类数。默认 ``1`` 。
        - **num_sampled** (int，可选) - 随机采样的类数。默认 ``5`` 。
        - **unique** (bool，可选) - 确认批处理中的所有采样类是否都是唯一的。如果 `unique` 为 ``True`` ，则批处理中的所有采样类都唯一。默认 ``True`` 。
        - **range_max** (int，可选) - 可能的类数。当 `unique` 为 ``True`` 时， `range_max` 必须大于或等于 `num_sampled` 。默认 ``5`` 。
        - **seed** (int，可选) - 随机种子，必须是非负。默认 ``0`` 。

    返回：
        3个Tensor组成的元组。

        - **sampled_candidates** (Tensor) - shape为 :math:`(num\_sampled,)` 且数据类型与 `true_classes` 相同的Tensor。
        - **true_expected_count** (Tensor) - shape与 `true_classes` 相同且数据类型为float32的Tensor。
        - **sampled_expected_count** (Tensor) - shape与 `sampled_candidates` 相同且数据类型为float32的Tensor。

    异常：
        - **TypeError** - `num_true` 和 `num_sampled` 都不是int。
        - **TypeError** - `unique` 不是bool。
        - **TypeError** - `range_max` 和 `seed` 都不是int。
        - **TypeError** - `true_classes` 不是Tensor。
