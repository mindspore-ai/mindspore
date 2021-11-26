mindspore.dataset.compare
==========================

.. py:method:: mindspore.dataset.compare(pipeline1, pipeline2)

    比较两个数据处理管道是否相同。

    **参数：**

    - **pipeline1** (Dataset)：数据处理管道。
    - **pipeline2** (Dataset)：数据处理管道。

    **返回：**

    bool，两个数据处理管道是否相等。

    **样例：**

    >>> pipeline1 = ds.MnistDataset(mnist_dataset_dir, 100)
    >>> pipeline2 = ds.Cifar10Dataset(cifar_dataset_dir, 100)
    >>> ds.compare(pipeline1, pipeline2)
    