mindspore.ops.LogUniformCandidateSampler
=========================================

.. py:class:: mindspore.ops.LogUniformCandidateSampler(num_true=1, num_sampled=5, unique=True, range_max=5, seed=0)

    使用log-uniform(Zipfian)分布对一组类别进行采样。

    该操作从整数范围[0, `range_max` )中随机采样一个采样类( `sampled_candidates` )的Tensor。

    更多参考详见 :func:`mindspore.ops.log_uniform_candidate_sampler`。

    参数：
        - **num_true** (int，可选) - 每个训练样本的目标类数。默认值： ``1`` 。
        - **num_sampled** (int，可选) - 随机采样的类数。默认值： ``5`` 。
        - **unique** (bool，可选) - 确认批处理中的所有采样类是否都是唯一的。如果 `unique` 为 ``True`` ，则批处理中的所有采样类都唯一。默认值： ``True`` 。
        - **range_max** (int，可选) - 可能的类数。当 `unique` 为 ``True`` 时， `range_max` 必须大于或等于 `num_sampled` 。默认值： ``5`` 。
        - **seed** (int，可选) - 随机种子，必须是非负。默认值： ``0`` 。

    输入：
        - **true_classes** (Tensor) - 目标类，其数据类型为int64，shape为 :math:`(batch\_size, num\_true)` 。

    输出：
        3个Tensor组成的元组。

        - **sampled_candidates** (Tensor) - shape为 :math:`(num\_sampled,)` 且数据类型与 `true_classes` 相同的Tensor。
        - **true_expected_count** (Tensor) - shape与 `true_classes` 相同且数据类型为float32的Tensor。
        - **sampled_expected_count** (Tensor) - shape与 `sampled_candidates` 相同且数据类型为float32的Tensor。
