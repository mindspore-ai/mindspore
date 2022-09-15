mindspore.dataset.PaddedDataset
================================

.. py:class:: mindspore.dataset.PaddedDataset(padded_samples)

    由用户提供的填充数据构建数据集。可用于在分布式训练时给原始数据集添加样本，使数据集样本能平均分配给不同的分片。

    参数：
        - **padded_samples** (list(dict)) - 用户提供的样本数据。

    异常：
        - **TypeError** - `padded_samples` 的类型不为list。
        - **TypeError** - `padded_samples` 的元素类型不为dict。
        - **ValueError** - `padded_samples` 为空的list。


.. include:: mindspore.dataset.api_list_nlp.rst
