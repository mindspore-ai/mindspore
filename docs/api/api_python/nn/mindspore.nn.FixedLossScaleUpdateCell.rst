mindspore.nn.FixedLossScaleUpdateCell
=======================================

.. py:class:: mindspore.nn.FixedLossScaleUpdateCell(loss_scale_value)

    固定梯度放大系数的神经元。

    该类是 :class:`mindspore.nn.FixedLossScaleManager` 的 `get_update_cell` 方法的返回值。训练过程中，类 :class:`mindspore.TrainOneStepWithLossScaleCell` 会调用该Cell。

    **参数：**

    - **loss_scale_value** (float) - 初始梯度放大系数。

    **输入：**

    - **loss_scale** (Tensor) - 训练期间的梯度放大系数，shape为 :math:`()`，在当前类中，该值被忽略。
    - **overflow** (bool) - 是否发生溢出。

    **输出：**

    Bool，即输入 `overflow`。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import numpy as np
    >>> from mindspore import Tensor, Parameter, nn, ops
    >>>
    >>> class Net(nn.Cell):
    ...     def __init__(self, in_features, out_features)：
    ...         super(Net, self).__init__()
    ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
    ...                                 name='weight')
    ...         self.matmul = ops.MatMul()
    ...
    ...     def construct(self, x)：
    ...         output = self.matmul(x, self.weight)
    ...         return output
    ...
    >>> in_features, out_features = 16, 10
    >>> net = Net(in_features, out_features)
    >>> loss = nn.MSELoss()
    >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> net_with_loss = nn.WithLossCell(net, loss)
    >>> manager = nn.FixedLossScaleUpdateCell(loss_scale_value=2**12)
    >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
    >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
    >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
    >>> output = train_network(input, labels)


    .. py:method:: get_loss_scale()

        获取当前梯度放大系数。
