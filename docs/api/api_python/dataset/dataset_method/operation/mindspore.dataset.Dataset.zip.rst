mindspore.dataset.Dataset.zip
=============================

.. py:method:: mindspore.dataset.Dataset.zip(datasets)

    将多个dataset对象按列进行合并压缩，多个dataset对象不能有相同的列名。

    参数：
        - **datasets** (tuple[Dataset]) - 要合并的（多个）dataset对象。

    返回：
        ZipDataset，合并后的dataset对象。

    异常：
        - **TypeError** - `datasets` 参数不是dataset对象/tuple(dataset)。
