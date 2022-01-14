.. py:class:: mindspore.train.callback.ModelCheckpoint(prefix='CKP', directory=None, config=None)

    checkpoint的回调函数。

    在训练过程中调用该方法可以保存网络参数。

    .. note::
        在分布式训练场景下，请为每个训练进程指定不同的目录来保存checkpoint文件。否则，可能会训练失败。

    **参数：**

    - **prefix** (str) - checkpoint文件的前缀名称。默认值：CKP。
    - **directory** (str) - 保存checkpoint文件的文件夹路径。默认情况下，文件保存在当前目录下。默认值：None。
    - **config** (CheckpointConfig) - checkpoint策略配置。默认值：None。

    **异常：**

    - **ValueError** - 如果prefix参数不是str类型或包含'/'字符。
    - **ValueError** - 如果directory参数不是str类型。
    - **TypeError** - config不是CheckpointConfig类型。

    **样例：**

    >>> from mindspore import Model, nn
    >>> from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
    >>>
    >>> class LeNet5(nn.Cell):
    ...     def __init__(self):
    ...         super(LeNet5, self).__init__()
    ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
    ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
    ...         self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
    ...         self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
    ...         self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
    ...         self.relu = nn.ReLU()
    ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    ...         self.flatten = nn.Flatten()
    ...
    ...     def construct(self, x)：
    ...         x = self.max_pool2d(self.relu(self.conv1(x)))
    ...         x = self.max_pool2d(self.relu(self.conv2(x)))
    ...         x = self.flatten(x)
    ...         x = self.relu(self.fc1(x))
    ...         x = self.relu(self.fc2(x))
    ...         x = self.fc3(x)
    ...         return x
    >>>
    >>> net = LeNet5()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    >>> optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    >>> model = Model(net, loss_fn=loss, optimizer=optim)
    >>> data_path = './MNIST_Data'
    >>> dataset = create_dataset(data_path)
    >>> config = CheckpointConfig(saved_network=net)
    >>> ckpoint_cb = ModelCheckpoint(prefix='LeNet5', directory='./checkpoint', config=config)
    >>> model.train(10, dataset, callbacks=ckpoint_cb)


    .. py:method:: end(run_context)

        在训练结束后，会保存最后一个step的checkpoint。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。

    .. py:method:: latest_ckpt_file_name
        :property:

        返回最新的checkpoint路径和文件名。

    .. py:method:: step_end(run_context)

        在step结束时保存checkpoint。

        **参数：**

        - **run_context** (RunContext) - 包含模型的一些基本信息。
