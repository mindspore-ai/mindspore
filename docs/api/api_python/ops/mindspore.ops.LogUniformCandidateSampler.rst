mindspore.ops.LogUniformCandidateSampler
=========================================

.. py:class:: mindspore.ops.LogUniformCandidateSampler(num_true=1, num_sampled=5, unique=True, range_max=5, seed=0)

    使用log-uniform(Zipfian)分布对一组类别进行采样。

    该操作从整数范围[0, `range_max` )中随机采样一个采样类( `sampled_candidates` )的Tensor。

    更多参考详见 :func:`mindspore.ops.log_uniform_candidate_sampler`。