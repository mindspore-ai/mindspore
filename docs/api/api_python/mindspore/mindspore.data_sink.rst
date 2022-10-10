mindspore.data_sink
===================

.. py:function:: mindspore.data_sink(fn, dataset, steps, sink_size=1, jit=False)

    对输入的函数封装生成一个新的函数。

    该生成的函数会以数据下沉模式执行。

    参数：
        - **fn** (Function) - 将与数据集一起运行的函数。
        - **dataset** (Dataset) - 训练数据集迭代器。数据集可以由数据集生成器API在 :class:`mindspore.dataset` 中生成，例如 :class:`mindspore.dataset.ImageFolderDataset` 。
        - **steps** (int) - 总的运行次数。 `steps` 必须为正整数。
        - **sink_size** (int) - 控制每次下沉的数据执行次数。 `sink_size` 必须为正整数。默认值：1。
        - **jit** (bool) - 控制生成函数的执行模式(graph模式/pynative模式)。默认值：False，采用pynative模式执行。

    返回：
        函数，该生成的函数会以数据下沉模式执行。

    异常：
        - **ValueError** - 如果 `steps` 或者 `sink_size` 不是正整数。
