mindspore.ops.UniformCandidateSampler
======================================

.. py:class:: mindspore.ops.UniformCandidateSampler(num_true, num_sampled, unique, range_max, seed=0, remove_accidental_hits=False)

    使用均匀分布对一组类别进行采样。

    此函数使用均匀分布从[0, range_max-1]中采样一组类（sampled_candidates）。如果 `unique` 为 ``True`` ，则候选采样没有重复；如果 `unique` 为 ``False`` ，则有重复。

    更多参考详见 :func:`mindspore.ops.uniform_candidate_sampler`。

    参数：
        - **num_true** (int) - 每个训练样本的目标类数。
        - **num_sampled** (int) - 随机采样的类数。sampled_candidates的shape将为 `num_sampled` 。如果 `unique` 为 ``True`` ，则 `num_sampled` 必须小于或等于 `range_max` 。
        - **unique** (bool) - 表示一个batch中的所有采样类是否唯一。
        - **range_max** (int) - 可能的类数，该值必须是非负的。
        - **seed** (int，可选) - 随机种子，该值必须是非负的。如果 `seed` 的值为 ``0`` ，则 `seed` 的值将被随机生成的值替换。默认值： ``0`` 。
        - **remove_accidental_hits** (bool，可选) - 表示是否移除accidental hit。accidental hit表示其中一个 `true_classes` 目标类匹配 `sampled_candidates` 采样类之一，设置为 ``True`` 表示移除等于目标类的采样类。默认值： ``False`` 。

    输入：
        - **true_classes** (Tensor) - 输入Tensor，目标类，其shape为 :math:`(batch\_size, num\_true)`。

    输出：
        - **sampled_candidates** (Tensor) -  候选采样与目标类之间不存在联系，其shape为 :math:`(num\_sampled, )`。
        - **true_expected_count** (Tensor) - 在每组目标类的采样分布下的预期计数。Shape为 :math:`(batch\_size, num\_true)`。
        - **sampled_expected_count** (Tensor) - 每个候选采样分布下的预期计数。Shape为 :math:`(num\_sampled, )`。
