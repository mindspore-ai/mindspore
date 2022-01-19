.. py:function:: mindspore.dataset.zip(datasets)

    将多个dataset对象按列进行合并压缩。

    **参数：**

    - **datasets** (tuple of class Dataset) - 输入元组格式的多个dataset对象。 `datasets` 参数的长度必须大于1。

    **返回：**

    ZipDataset，合并后的dataset对象。

    **异常：**

    - **ValueError** - datasets参数的长度为1。
    - **TypeError** - datasets参数不是元组。
