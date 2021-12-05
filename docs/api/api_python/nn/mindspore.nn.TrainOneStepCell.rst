mindspore.nn.TrainOneStepCell
=============================

.. py:class:: mindspore.nn.TrainOneStepCell(network, optimizer, sens=1.0)

    训练网络封装类。

    封装 `network` 和 `optimizer` ，构建一个输入'\*inputs'的用于训练的Cell。
    执行函数 `construct` 中会构建反向图以更新网络参数。支持不同的并行训练模式。

    **参数：**

    - **network** (Cell) - 训练网络。只支持单输出网络。
    - **optimizer** (Union[Cell]) - 用于更新网络参数的优化器。
    - **sens** (numbers.Number) - 反向传播的输入，缩放系数。默认值为1.0。

    **输入：**

    **(\*inputs)** (Tuple(Tensor)) - shape为 :math:`(N, \ldots)` 的Tensor组成的元组。

    **输出：**

    Tensor，损失函数值，其shape通常为 :math:`()` 。

    **异常：**

    **TypeError**：`sens` 不是numbers.Number。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    >>> optim = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> # 1）使用MindSpore提供的WithLossCell
    >>> loss_net = nn.WithLossCell(net, loss_fn)
    >>> train_net = nn.TrainOneStepCell(loss_net, optim)
    >>>
    >>> # 2）用户自定义的WithLossCell
    >>> class MyWithLossCell(Cell):
    ...    def __init__(self, backbone, loss_fn):
    ...        super(MyWithLossCell, self).__init__(auto_prefix=False)
    ...        self._backbone = backbone
    ...        self._loss_fn = loss_fn
    ...
    ...    def construct(self, x, y, label):
    ...        out = self._backbone(x, y)
    ...        return self._loss_fn(out, label)
    ...
    ...    @property
    ...    def backbone_network(self):
    ...        return self._backbone
    ...
    >>> loss_net = MyWithLossCell(net, loss_fn)
    >>> train_net = nn.TrainOneStepCell(loss_net, optim)
