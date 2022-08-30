mindspore.ops.uniform_candidate_sampler
=========================================

.. py:function:: mindspore.ops.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=0, remove_accidental_hits=False)

    使用均匀分布对一组类别进行采样。

    此函数使用均匀分布从[0, range_max-1]中采样一组类（sampled_candidates）。如果 `unique` 为True，则候选采样没有重复；如果 `unique` 为False，则有重复。

    参数：
        - **true_classes** (Tensor) - 输入Tensor，目标类，其shape为(batch_size, num_true)。
        - **num_true** (int) - 每个训练样本的目标类数。
        - **num_sampled** (int) - 随机采样的类数。sampled_candidates的shape将为 `num_sampled` 。如果 `unique` 为True，则 `num_sampled` 必须小于或等于 `range_max` 。
        - **unique** (bool) - 表示一个batch中的所有采样类是否唯一。
        - **range_max** (int) - 可能的类数，该值必须大于0。
        - **seed** (int) - 随机种子，该值必须是非负的。如果seed的值为0，则seed的值将被随机生成的值替换。默认值：0。
        - **remove_accidental_hits** (bool) - 表示是否移除accidental hit。默认值：False。

    返回：
        - **sampled_candidates** (Tensor) -  候选采样与目标类之间不存在联系，其shape为(num_sampled, )。
        - **true_expected_count** (Tensor) - 在每组目标类的采样分布下的预期计数。Shape为(batch_size, num_true)。
        - **sampled_expected_count** (Tensor) - 每个候选采样分布下的预期计数。Shape为(num_sampled, )。

    异常：
        - **TypeError** - `num_true` 和 `num_sampled` 都不是int。
        - **TypeError** - `uique` 和 `remo_acidental_hits` 都不是bool。
        - **TypeError** - `range_max` 和 `seed` 都不是int。
        - **TypeError** - `true_classes` 不是Tensor。
