mindspore.dataset.PKSampler
==============================

.. py:class:: mindspore.dataset.PKSampler(num_val, num_class=None, shuffle=False, class_column='label', num_samples=None)

    为数据集中每P个类别各采样K个样本。

    参数：
        - **num_val** (int) - 每个类要采样的元素数量。
        - **num_class** (int, 可选) - 要采样的类数量。默认值为 ``None`` ，采样所有类。当前不支持指定该参数。
        - **shuffle** (bool, 可选) - 是否混洗采样得到的样本。默认值： ``False`` ，不混洗样本。
        - **class_column** (str, 可选) - 指定label所属数据列的名称，将基于此列作为数据标签进行采样。默认值： ``'label'`` 。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。

    异常：
        - **TypeError** - `shuffle` 的类型不是bool。
        - **TypeError** - `class_column` 的类型不是str。
        - **TypeError** - `num_samples` 的类型不是int。
        - **NotImplementedError** - `num_class` 不为 ``None`` 。
        - **RuntimeError** - `num_val` 不是正值。
        - **ValueError** - `num_samples` 为负值。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst