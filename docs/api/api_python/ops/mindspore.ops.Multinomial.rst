mindspore.ops.Multinomial
==========================

.. py:class:: mindspore.ops.Multinomial(seed=0, seed2=0, dtype=mstype.int32)

    返回从输入Tensor对应行进行多项式概率分布采样出的Tensor。

    .. note::
        输入的行不需要求和为1(在这种情况下，我们使用值作为权重)，但必须是非负的、有限的，并且具有非零和。

    参数：
        - **seed** (int) - 随机数种子，必须是非负数。默认值：0。
        - **seed2** (int) - 二号随机数种子，必须是非负数。默认值：0。
        - **dtype** (dtype) - 输出数据类型，必须是int32或者int64，默认类型：int32。

    输入：
        - **x** (Tensor) - 包含累加概率和的输入Tensor，必须是一维或二维。CPU和GPU后端支持一维或者二维，Ascend后端仅支持二维。
        - **num_samples** (int) - 要抽取的样本数。

    输出：
        Tensor，具有与输入相同的行。每行的采样索引数为 `num_samples` 。

    异常：
        - **TypeError** - 如果 `seed` 或者 `seed2` 不是int类型。
        - **TypeError** - 如果 `num_sample` 不是int类型。
        - **TypeError** - 如果 `dtype` 不是int32或者int64类型。
        - **ValueError** - 如果 `seed` 或者 `seed2` 小于零。
