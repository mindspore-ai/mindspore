mindspore.nn.DistributedGradReducer
===================================

.. py:class:: mindspore.nn.DistributedGradReducer(parameters, mean=True, degree=None, fusion_type=1, group=GlobalComm.WORLD_COMM_GROUP)

    分布式优化器。

    对反向梯度进行AllReduce运算。

    **参数：**

    - **parameters** (list) - 需要更新的参数。
    - **mean** (bool) - 当mean为True时，对AllReduce之后的梯度求均值。默认值：False。
    - **degree** (int) - 平均系数，通常等于设备编号。默认值：None。
    - **fusion_type** (int) - AllReduce算子的融合类型。默认值：1。

    **异常：**

    **ValueError**：如果degree不是int或小于0。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> #此示例应与多个进程一起运行。
    >>> #请参考Mindpore.cn上的“教程>分布式训练”。
    >>> import numpy as np
    >>> from mindspore.communication import init
    >>> from mindspore import ops
    >>> from mindspore import context
    >>> from mindspore.context import ParallelMode
    >>> from mindspore import Parameter, Tensor
    >>> from mindspore import nn
    >>>
    >>> context.set_context(mode=context.GRAPH_MODE)
    >>> init()
    >>> context.reset_auto_parallel_context()
    >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
    >>>
    >>> class TrainingWrapper(nn.Cell):
    ...     def __init__(self, network, optimizer, sens=1.0):
    ...         super(TrainingWrapper, self).__init__(auto_prefix=False)
    ...         self.network = network
    ...         self.network.add_flags(defer_inline=True)
    ...         self.weights = optimizer.parameters
    ...         self.optimizer = optimizer
    ...         self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
    ...         self.sens = sens
    ...         self.reducer_flag = False
    ...         self.grad_reducer = None
    ...         self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
    ...         if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
    ...             self.reducer_flag = True
    ...         if self.reducer_flag:
    ...             mean = context.get_auto_parallel_context("gradients_mean")
    ...             degree = context.get_auto_parallel_context("device_num")
    ...             self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    ...
    ...     def construct(self, *args):
    ...         weights = self.weights
    ...         loss = self.network(*args)
    ...         sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
    ...         grads = self.grad(self.network, weights)(*args, sens)
    ...         if self.reducer_flag:
    ...             # apply grad reducer on grads
    ...             grads = self.grad_reducer(grads)
    ...         return ops.Depend(loss, self.optimizer(grads))
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
    >>>
    >>> size, in_features, out_features = 16, 16, 10
    >>> network = Net(in_features, out_features)
    >>> loss = nn.MSELoss()
    >>> net_with_loss = nn.WithLossCell(network, loss)
    >>> optimizer = nn.Momentum(net_with_loss.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> train_cell = TrainingWrapper(net_with_loss, optimizer)
    >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
    >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
    >>> grads = train_cell(inputs, label)
    >>> print(grads)
    256.0

.. py:method:: construct(grads)

    某些情况下，梯度的数据精度可以与float16和float32混合。因此，AllReduce的结果不可靠。要解决这个问题，必须在AllReduce之前强制转换为float32，并在操作之后再强制转换为float32。

    **参数：**

    - **grads** (Union[Tensor, tuple[Tensor]]) - 操作前的梯度Tensor或tuple。

    **返回：**

    - **new_grads** (Union[Tensor, tuple[Tensor]])，操作后的梯度Tensor或tuple。
