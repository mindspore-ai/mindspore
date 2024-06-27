mindspore.data_sink
===================

.. py:function:: mindspore.data_sink(fn, dataset, sink_size=1, jit_config=None, input_signature=None)

    对输入的函数封装生成一个新的函数。

    .. note::
        使用数据下沉时，数据集将被自动循环发送至设备，设备侧最多缓存100个batch的数据且所占内存不大于2G，此时仅需考虑每次下沉的步数 `sink_size` ， `sink_size` 默认为 ``1`` ，代表每个epoch仅从缓存中取一个batch的数据进行训练并输出loss，若 `sink_size` 大于1，则每个epoch从缓存中取出 `sink_size` 个batch的数据进行训练然后输出loss。

    参数：
        - **fn** (Function) - 将与数据集一起运行的函数。
        - **dataset** (Dataset) - 训练数据集迭代器。数据集可以由数据集生成器API在 `mindspore.dataset` 中生成，例如 :class:`mindspore.dataset.ImageFolderDataset` 。
        - **sink_size** (int) - 控制每次数据下沉的step数量。 `sink_size` 必须为正整数。默认值： ``1`` 。
        - **jit_config** (JitConfig) - 编译时所使用的JitConfig配置项，详细可参考 :class:`mindspore.JitConfig` 。默认值： ``None`` ，表示以PyNative模式运行。
        - **input_signature** (Union[Tensor, List or Tuple of Tensors]) - 用于表示输入参数的Tensor。Tensor的shape和dtype将作为函数的输入shape和dtype。如果指定了 `input_signature` ，则 `fn` 的每个输入都必须是Tensor，并且 `fn` 的输入参数将不会接受 `**kwargs` 参数，实际输入的shape和dtype应与 `input_signature` 相同，否则会出现TypeError。默认值： ``None`` 。

    返回：
        函数，该生成的函数会以数据下沉模式执行。

    异常：
        - **ValueError** - 如果 `sink_size` 不是正整数。