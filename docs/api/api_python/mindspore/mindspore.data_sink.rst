mindspore.data_sink
===================

.. py:function:: mindspore.data_sink(fn, dataset, steps, sink_size=1, jit_config=None, input_signature=None)

    对输入的函数封装生成一个新的函数。

    该生成的函数会以数据下沉模式执行。

    参数：
        - **fn** (Function) - 将与数据集一起运行的函数。
        - **dataset** (Dataset) - 训练数据集迭代器。数据集可以由数据集生成器API在 :class:`mindspore.dataset` 中生成，例如 :class:`mindspore.dataset.ImageFolderDataset` 。
        - **steps** (int) - 总的运行次数。 `steps` 必须为正整数。
        - **sink_size** (int) - 控制每次下沉的数据执行次数。 `sink_size` 必须为正整数。默认值：1。
        - **jit_config** (JitConfig) - 编译时所使用的JitConfig配置项，详细可参考 :class:`mindspore.JitConfig` 。默认值：None。
        - **input_signature** (Tensor) - 用于表示输入参数的Tensor。Tensor的shape和dtype将作为函数的输入shape和dtype。默认值：None。

    返回：
        函数，该生成的函数会以数据下沉模式执行。

    异常：
        - **ValueError** - 如果 `steps` 或者 `sink_size` 不是正整数。