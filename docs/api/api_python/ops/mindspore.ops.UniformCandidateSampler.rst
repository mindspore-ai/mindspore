mindspore.ops.UniformCandidateSampler
======================================

.. py:class:: mindspore.ops.UniformCandidateSampler(num_true, num_sampled, unique, range_max, seed=0, remove_accidental_hits=False)

    使用均匀分布对一组类别进行采样。

    此函数使用均匀分布从[0, range_max-1]中采样一组类（sampled_candidates）。如果 `unique` 为True，则候选采样没有重复；如果 `unique` 为False，则有重复。

    更多参考详见 :func:`mindspore.ops.uniform_candidate_sampler`。
