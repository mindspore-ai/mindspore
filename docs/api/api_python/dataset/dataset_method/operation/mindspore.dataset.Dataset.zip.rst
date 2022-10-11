mindspore.dataset.Dataset.zip
=============================

.. py:method:: mindspore.dataset.Dataset.zip(datasets)

    将多个dataset对象按列进行合并压缩。

    参数：
        - **datasets** (tuple[Dataset]) - 要合并的多个dataset对象。 `datasets` 参数的长度必须大于1。

    返回：
        ZipDataset，合并后的dataset对象。

    异常：
        - **ValueError** - `datasets` 参数的长度为1。
        - **TypeError** - `datasets` 参数不是tuple。
