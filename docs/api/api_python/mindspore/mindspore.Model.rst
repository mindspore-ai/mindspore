mindspore.Model
================

.. py:class:: mindspore.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", acc_level="O0", **kwargs)

    模型训练或推理的高阶接口。 `Model` 会根据用户传入的参数封装可训练或推理的实例。

    **参数：**

    - **network** (Cell) – 用于训练或推理的神经网络。
    - **loss_fn** (Cell) - 损失函数。如果 `loss_fn` 为None，`network` 中需要进行损失函数计算，必要时也需要进行并行计算。默认值：None。
    - **optimizer** (Cell) - 用于更新网络权重的优化器。如果 `optimizer` 为None， `network` 中需要进行反向传播和网络权重更新。默认值：None。
    - **metrics** (Union[dict, set]) - 用于模型评估的一组评价函数。例如：{'accuracy', 'recall'}。默认值：None。
    - **eval_network** (Cell) - 用于评估的神经网络。未定义情况下，`Model` 会使用 `network` 和 `loss_fn` 封装一个 `eval_network` 。默认值：None。
    - **eval_indexes** (list) - 在定义 `eval_network` 的情况下使用。如果 `eval_indexes` 为默认值None，`Model` 会将 `eval_network` 的所有输出传给 `metrics` 。如果配置 `eval_indexes` ，必须包含三个元素，分别为损失值、预测值和标签在 `eval_network` 输出中的位置，此时，损失值将传给损失评价函数，预测值和标签将传给其他评价函数。推荐使用评价函数的 `mindspore.nn.Metric.set_indexes` 代替 `eval_indexes` 。默认值：None。
    - **amp_level** (str) - `mindspore.build_train_network` 的可选参数 `level`，`level` 为混合精度等级，该参数支持["O0", "O2", "O3", "auto"]。默认值："O0"。

      - O0: 无变化。
      - O2: 将网络精度转为float16，batchnorm保持float32精度，使用动态调整梯度放大系数（loss scale）的策略。
      - O3: 将网络精度（包括batchnorm）转为float16，不使用梯度调整策略。
      - auto: 为不同处理器设置专家推荐的混合精度等级，如在GPU上设为O2，在Ascend上设为O3。该设置方式可能在部分场景下不适用，建议用户根据具体的网络模型自定义设置 `amp_level` 。

      在GPU上建议使用O2，在Ascend上建议使用O3。
      通过`kwargs`设置`keep_batchnorm_fp32`，可修改batchnorm策略，`keep_batchnorm_fp32`必须为bool类型；通过`kwargs`设置`loss_scale_manager`可修改梯度放大策略，`loss_scale_manager`必须为:class:`mindspore.LossScaleManager`的子类，
      关于 `amp_level` 详见 `mindpore.build_train_network`。

    **样例:**

    >>> from mindspore import Model, nn
    >>>
    >>> class Net(nn.Cell):
    ...     def __init__(self, num_class=10, num_channel=1):
    ...         super(Net, self).__init__()
    ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
    ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
    ...         self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
    ...         self.fc2 = nn.Dense(120, 84, weight_init='ones')
    ...         self.fc3 = nn.Dense(84, num_class, weight_init='ones')
    ...         self.relu = nn.ReLU()
    ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    ...         self.flatten = nn.Flatten()
    ...
    ...     def construct(self, x):
    ...         x = self.max_pool2d(self.relu(self.conv1(x)))
    ...         x = self.max_pool2d(self.relu(self.conv2(x)))
    ...         x = self.flatten(x)
    ...         x = self.relu(self.fc1(x))
    ...         x = self.relu(self.fc2(x))
    ...         x = self.fc3(x)
    ...         return x
    >>>
    >>> net = Net()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    >>> # 如何构建数据集，请参考官方网站的数据集相关章节
    >>> dataset = create_custom_dataset()
    >>> model.train(2, dataset)

    .. py:method:: build(train_dataset=None, valid_dataset=None, sink_size=-1)

        数据下沉模式下构建计算图和数据图。

        .. warning::这是一个实验性接口，后续可能删除或修改。

        .. note:: 如果预先调用该接口构建计算图，那么 `Model.train` 会直接执行计算图。预构建计算图目前仅支持GRAPH_MOD模式和Ascend处理器，仅支持数据下沉模式。

        **参数：**

        - **train_dataset** (Dataset) – 一个训练集迭代器。如果定义了 `train_dataset` ，将会构建训练计算图。默认值：None。
        - **valid_dataset** (Dataset) - 一个验证集迭代器。如果定义了 `valid_dataset` ，将会构建验证计算图，此时 `Model` 中的 `metrics` 不能为None。默认值：None。
        - **sink_size** (int) - 控制每次数据下沉的数据量。默认值：-1。
        - **epoch** (int) - 控制训练轮次。默认值：1。

        **样例：**

        >>> from mindspore import Model, nn, FixedLossScaleManager
        >>>
        >>> # 如何构建数据集，请参考官方网站的数据集相关章节
        >>> dataset = create_custom_dataset()
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_scale_manager = FixedLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
        >>> model.build(dataset, epoch=2)
        >>> model.train(2, dataset)
        >>> model.train(2, dataset)

    .. py:method:: eval(valid_dataset, callbacks=None, dataset_sink_mode=True)

        模型评估接口。

        使用PyNative模式或CPU处理器时，模型评估流程将以非下沉模式执行。

        .. note::
            如果 `dataset_sink_mode` 配置为True，数据将被送到处理器中。如果处理器是Ascend，数据特征将被逐一传输，每次数据传输的上限是256M。如果 `dataset_sink_mode` 配置为True，数据集仅能在当前模型中使用。该接口会构建并执行计算图，如果使用前先执行了 `Model.build` ，那么它会直接执行计算图而不构建。

        **参数：**

        - **valid_dataset** (Dataset) – 评估模型的数据集。
        - **callbacks** (Optional[list(Callback), Callback]) - 评估过程中需要执行的回调对象或回调对象列表。默认值：None。
        - **dataset_sink_mode** (bool) - 数据是否直接下沉至处理器进行处理。默认值：True。

        **返回：**

        Dict，键是用户定义的评价指标名称，值是以推理模式运行的评估结果。

        **样例：**

        >>> from mindspore import Model, nn
        >>>
        >>> # 如何构建数据集，请参考官方网站的数据集相关章节
        >>> dataset = create_custom_dataset()
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
        >>> acc = model.eval(dataset, dataset_sink_mode=False)

    .. py:method:: eval_network
        :property:

        获取该模型的评价网络。

        **返回：**

        评估网络实例。

    .. py:method:: infer_predict_layout(*predict_data)

        在 `AUTO_PARALLEL` 或 `SEMI_AUTO_PARALLEL` 模式下为预测网络生成参数layout，数据可以是单个或多个张量。

        .. note:: 同一批次数据应放在一个张量中。

        **参数：**

        - **predict_data** (Tensor) – 单个或多个张量的预测数据。

        **返回：**

        Dict，用于加载分布式checkpoint的参数layout字典。它总是作为 `load_distributed_checkpoint()` 函数的一个入参。

        **异常：**

        - **RuntimeError** – 非图模式（GRAPH_MODE）将会抛出该异常。

        **样例：**

        >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Model, context, Tensor
        >>> from mindspore.context import ParallelMode
        >>> from mindspore.communication import init
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> init()
        >>> context.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
        >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
        >>> model = Model(Net())
        >>> model.infer_predict_layout(input_data)

    .. py:method:: infer_train_layout(train_dataset, dataset_sink_mode=True, sink_size=-1)

        在 `AUTO_PARALLEL` 或 `SEMI_AUTO_PARALLEL` 模式下为训练网络生成参数layout，当前仅支持在数据下沉模式下使用。

        .. warning:: 这是一个实验性的原型，可能会被改变和/或删除。

        .. note:: 这是一个预编译函数。参数必须与Model.train()函数相同。

        **参数：**

        - **train_dataset** (Dataset) – 一个训练数据集迭代器。如果没有损失函数（loss_fn），返回一个包含多个数据的元组（data1, data2, data3, ...）并传递给网络。否则，返回一个元组（data, label），数据和标签将被分别传递给网络和损失函数。
        - **dataset_sink_mode** (bool) – 决定是否以数据集下沉模式进行训练。默认值：True。PyNative模式下或处理器为CPU时，训练模型流程使用的是数据不下沉（non-sink）模式。默认值：True。
        - **sink_size** (int) – 控制每次数据下沉的数据量，如果 `sink_size` =-1，则每一次epoch下沉完整数据集。如果 `sink_size` >0，则每一次epoch下沉数据量为 `sink_size` 的数据集。如果 `dataset_sink_mode` 为False，则设置 `sink_size` 为无效。默认值：-1。

        **返回：**

        Dict，用于加载分布式checkpoint的参数layout字典。

        **样例：**

        >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Model, context, Tensor, nn, FixedLossScaleManager
        >>> from mindspore.context import ParallelMode
        >>> from mindspore.communication import init
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> init()
        >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
        >>>
        >>> # 如何构建数据集，请参考官方网站上关于[数据集]的章节。
        >>> dataset = create_custom_dataset()
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_scale_manager = FixedLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
        >>> layout_dict = model.infer_train_layout(dataset)

    .. py:method:: predict(*predict_data)

        输入样本得到预测结果。

        **参数：**

        - **predict_data** (Tensor) – 预测样本，数据可以是单个张量、张量列表或张量元组。

        **返回：**

        返回预测结果，类型是Tensor或Tensor元组。

        **样例:**

        >>> import mindspore as ms
        >>> from mindspore import Model, Tensor
        >>>
        >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
        >>> model = Model(Net())
        >>> result = model.predict(input_data)

    .. py:method:: predict_network
        :property:

        获得该模型的预测网络。

        **返回：**

        预测网络实例。

    .. py:method:: train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1)

        模型训练接口。

        使用PYNATIVE_MODE模式或CPU处理器时，模型训练流程将以非下沉模式执行。

        .. note::
            如果 `dataset_sink_mode` 配置为True，数据将被送到处理器中。如果处理器是Ascend，数据特征将被逐一传输，每次数据传输的上限是256M。如果 `dataset_sink_mode` 配置为True，仅在每个epoch结束时调用Callback实例的step_end方法。如果 `dataset_sink_mode` 配置为True，数据集仅能在当前模型中使用。如果 `sink_size` 大于零，每次epoch可以无限次遍历数据集，直到遍历数据量等于 `sink_size` 为止。每次epoch将从上一次遍历的最后位置继续开始遍历。该接口会构建并执行计算图，如果使用前先执行了 `Model.build` ，那么它会直接执行计算图而不构建。

        **参数：**

        - **epoch** (int) – 训练执行轮次。通常每个epoch都会使用全量数据集进行训练。当 `dataset_sink_mode` 设置为True且 `sink_size` 大于零时，则每个epoch训练次数为 `sink_size` 而不是数据集的总步数。
        - **train_dataset** (Dataset) – 一个训练数据集迭代器。如果定义了 `loss_fn` ，则数据和标签会被分别传给 `network` 和 `loss_fn` ，此时数据集需要返回一个元组（data, label）。如果数据集中有多个数据或者标签，可以设置 `loss_fn` 为None，并在 `network` 中实现损失函数计算，此时数据集返回的所有数据组成的元组（data1, data2, data3, ...）会传给 `network` 。
        - **callback** (Optional[list[Callback], Callback]) – 训练过程中需要执行的回调对象或者回调对象列表。默认值：None。
        - **dataset_sink_mode** (bool) – 数据是否直接下沉至处理器进行处理。使用PYNATIVE_MODE模式或CPU处理器时，模型训练流程将以非下沉模式执行。默认值：True。
        - **sink_size** (int) – 控制每次数据下沉的数据量。`dataset_sink_mode` 为False时 `sink_size` 无效。如果sink_size=-1，则每一次epoch下沉完整数据集。如果sink_size>0，则每一次epoch下沉数据量为sink_size的数据集。默认值：-1。

        **样例:**

        >>> from mindspore import Model, nn, FixedLossScaleManager
        >>>
        >>> # 如何构建数据集，请参考官方网站的数据集相关章节
        >>> dataset = create_custom_dataset()
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> loss_scale_manager = FixedLossScaleManager()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
        >>> model.train(2, dataset)

    .. py:method:: train_network
        :property:

        获得该模型的训练网络。

        **返回：**

        预测网络实例。