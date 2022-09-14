mindspore.Tensor.random_categorical
===================================

.. py:method:: mindspore.Tensor.random_categorical(num_sample, seed=0, dtype=mstype.int64)

    从分类分布中抽取样本。

    参数：
        - **num_sample** (int): 要抽取的样本数。只允许使用常量值。
        - **seed** (int): 随机种子。只允许使用常量值。默认值：0。
        - **dtype** (mindspore.dtype): 输出的类型。它的值必须是 mindspore.int16、mindspore.int32 和 mindspore.int64 之一。默认值：mindspore.int64。

    返回：
        Tensor，Shape为 :math:`(batch\_size, num\_samples)` 的输出Tensor。

    异常：
        - **TypeError** - 如果 `dtype` 不是以下之一：mindspore.int16、mindspore.int32、mindspore.int64。
        - **TypeError** - 如果 `logits` 不是Tensor。
        - **TypeError** - 如果 `num_sample` 或者 `seed` 不是 int。