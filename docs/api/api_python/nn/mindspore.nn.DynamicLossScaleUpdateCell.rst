mindspore.nn.DynamicLossScaleUpdateCell
=======================================

.. py:class:: mindspore.nn.DynamicLossScaleUpdateCell(loss_scale_value, scale_factor, scale_window)

    用于动态地更新梯度放大系数(loss scale)的神经元。

    使用梯度放大功能进行训练时，初始梯度放大系数值为 `loss_scale_value`。在每个训练步骤中，当出现溢出时，通过计算公式 `loss_scale`/`scale_factor` 减小梯度放大系数。如果连续 `scale_window` 步（step）未溢出，则将通过 `loss_scale` * `scale_factor` 增大梯度放大系数。

    该类是 :class:`mindspore.nn.DynamicLossScaleManager` 的 `get_update_cell` 方法的返回值。训练过程中，类 :class:`mindspore.TrainOneStepWithLossScaleCell` 会调用该Cell来更新梯度放大系数。

    **参数：**

    - **loss_scale_value** (float) - 初始的梯度放大系数。
    - **scale_factor** (int) - 增减系数。
    - **scale_window** (int) - 未溢出时，增大梯度放大系数的最大连续训练步数。

    **输入：**

    - **loss_scale** (Tensor) - 训练期间的梯度放大系数，shape为 :math:`()`。
    - **overflow** (bool) - 是否发生溢出。

    **输出：**

    Bool，即输入 `overflow` 。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import numpy as np
    >>> from mindspore import Tensor, Parameter, nn
    >>> import mindspore.ops as ops
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
    >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    >>> train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
    >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
    >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
    >>> output = train_network(input, labels)


    .. py:method:: get_loss_scale()

        获取当前梯度放大系数。
