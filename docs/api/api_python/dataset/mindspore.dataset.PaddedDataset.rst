mindspore.dataset.PaddedDataset
================================

.. py:class:: mindspore.dataset.PaddedDataset(padded_samples)

    使用用户提供的填充数据创建数据集。可用于在分布式训练时给原始数据集添加样本，使数据集能平均分配给不同的分片。

    **参数：**

    - **padded_samples** (list(dict)): 用户提供的样本数据。

    **异常：**

    - **TypeError** - `padded_samples` 的类型不为list。
    - **TypeError** - `padded_samples` 的元素类型不为dict。
    - **ValueError** - `padded_samples` 为空列表。

    **样例：**

    >>> import numpy as np
    >>> data = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)}]
    >>> dataset = ds.PaddedDataset(padded_samples=data)

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
