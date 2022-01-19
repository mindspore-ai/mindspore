mindspore.dataset.WeightedRandomSampler
=======================================

.. py:class:: mindspore.dataset.WeightedRandomSampler(weights, num_samples=None, replacement=True)

    给定样本的权重列表，根据权重决定样本的采样概率，随机采样[0，len(weights) - 1]中的样本。

    **参数：**

    - **weights** (list[float, int]) - 权重序列，总和不一定为1。
    - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值：None，获取采样到的所有样本。
    - **replacement** (bool) - 是否将样本ID放回下一次采样，默认值：True，有放回采样。

    **异常：**

    - **TypeError：** `weights` 元素的类型不是数字。
    - **TypeError：** `num_samples` 不是整数值。
    - **TypeError：** `replacement` 不是布尔值。
    - **RuntimeError：** `weights` 为空或全为零。
    - **ValueError：** `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst