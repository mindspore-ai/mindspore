mindspore.dataset.RandomSampler
================================

.. py:class:: mindspore.dataset.RandomSampler(replacement=False, num_samples=None)

    随机采样器。

    参数：
        - **replacement** (bool, 可选) - 是否将样本ID放回下一次采样。默认值： ``False`` ，无放回采样。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。

    异常：
        - **TypeError** - `replacement` 不是bool值。
        - **TypeError** - `num_samples` 不是整数值。
        - **ValueError** - `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst