mindspore.ops.RandomCategorical
===============================

.. py:class:: mindspore.ops.RandomCategorical(dtype=mstype.int64)

    从分类分布中抽取样本。

    参数：
        - **dtype** (mindspore.dtype) - 输出的类型。它的值必须是 mstype.int16、mstype.int32 和 mstype.int64 之一。默认值： ``mstype.int64`` 。

    输入：
        - **logits** (Tensor) - 输入Tensor。Shape为 :math:`(batch\_size, num\_classes)` 的二维Tensor。
        - **num_sample** (int) - 要抽取的样本数。只允许使用常量值。
        - **seed** (int) - 随机种子。只允许使用常量值。默认值： ``0`` 。

    输出：
        - **output** (Tensor) - Shape为 :math:`(batch\_size, num\_samples)` 的输出Tensor。

    异常：
        - **TypeError** - 如果 `dtype` 不是以下之一：mstype.int16、mstype.int32、mstype.int64。
        - **TypeError** - 如果 `logits` 不是Tensor。
        - **TypeError** - 如果 `num_sample` 或者 `seed` 不是 int。
