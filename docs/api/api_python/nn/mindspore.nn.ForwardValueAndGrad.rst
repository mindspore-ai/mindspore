mindspore.nn.ForwardValueAndGrad
===================================

.. py:class:: mindspore.nn.ForwardValueAndGrad(network, weights=None, get_all=False, get_by_list=False, sens_param=False)

    训练网络的封装。

    包括正向网络和梯度函数。该类生成的Cell使用'\*inputs'输入来训练。
    通过梯度函数来创建反向图，用以计算梯度。

    **参数：**
    
    - **network** (Cell) - 训练网络。
    - **weights** (ParameterTuple) - 训练网络中需要计算梯度的的参数。
    - **get_all** (bool) - 如果为True，则计算网络输入对应的梯度。默认值：False。
    - **get_by_list** (bool) - 如果为True，则计算参数变量对应的梯度。如果 `get_all` 和 `get_by_list` 都为False，则计算第一个输入对应的梯度。如果 `get_all` 和 `get_by_list` 都为True，则以（（输入的梯度）,（参数的梯度））的形式同时获取输入和参数变量的梯度。默认值：False。
    - **sens_param** (bool) - 是否将sens作为输入。如果 `sens_param` 为False，则sens默认为'ones_like(outputs)'。默认值：False。如果 `sens_param` 为True，则需要指定sens的值。
            
    **输入：**
    
    - **(\*inputs)** (Tuple(Tensor...))：shape为 :math:`(N, \ldots)` 的输入tuple。
    - **(sens)**：反向传播梯度的缩放值。如果网络有单个输出，则sens是tensor。如果网络有多个输出，则sens是tuple(tensor)。

    **输出：**

    - **forward value** - 网络运行的正向结果。
    - **gradients** (tuple(tensor)) - 网络反向传播的梯度。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> import numpy as np
    >>> from mindspore import Tensor, nn, common, ops, ParameterTuple, Parameter
    >>>
    >>> class Net(nn.Cell):
    ...    def __init__(self):
    ...        super(Net, self).__init__()
    ...        self.weight = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="weight")
    ...        self.matmul = ops.MatMul()
    ...
    ...    def construct(self, x):
    ...        out = self.matmul(x, self.weight)
    ...        return out
    ...
    >>> net = Net()
    >>> criterion = nn.SoftmaxCrossEntropyWithLogits()
    >>> net_with_criterion = nn.WithLossCell(net, criterion)
    >>> weight = ParameterTuple(net.trainable_params())
    >>> train_network = nn.ForwardValueAndGrad(net_with_criterion, weights=weight, get_all=True, get_by_list=True)
    >>> inputs = Tensor(np.ones([1, 2]).astype(np.float32))
    >>> labels = Tensor(np.zeros([1, 2]).astype(np.float32))
    >>> result = train_network(inputs, labels)
    >>> print(result)
    (Tensor(shape=[1], dtype=Float32, value=[0.00000000e+00]), ((Tensor(shape=[1, 2], dtype=Float32, value=
    [[1.00000000e+00, 1.00000000e+00]]), Tensor(shape=[1, 2], dtype=Float32, value=
    [[0.00000000e+00, 0.00000000e+00]])), (Tensor(shape=[2, 2], dtype=Float32, value=
    [[5.00000000e-01, 5.00000000e-01],
     [5.00000000e-01, 5.00000000e-01]]),)))
    