mindspore.Parameter
========================

.. py:class:: mindspore.Parameter(default_input, *args, **kwargs)

    通常表示网络的参数（ `Parameter` 是 `Tensor` 的子类）。

    .. note::
        在"semi_auto_parallel"和"auto_parallel"的并行模式下，如果使用 `Initializer` 模块初始化参数，参数的类型将为 `Tensor` ，:class:`mindspore.ops.AllGather`
        `Tensor` 仅保存张量的形状和类型信息，而不占用内存来保存实际数据。并行场景下存在参数的形状发生变化的情况，用户可以调用 `Parameter` 的 `init_data` 方法得到原始数据。
        如果网络中存在需要部分输入为 `Parameter` 的算子，则不允许这部分输入的 `Parameter` 进行转换。
        如果在 `Cell` 里初始化一个 `Parameter` 作为 `Cell` 的属性时，建议使用默认值None，否则 `Parameter` 的 `name` 可能与预期不一致。

    **参数：**

        - **default_input** (Union[Tensor, int, float, numpy.ndarray, list])：初始化参数的输入值。
        - **name** (str)：参数的名称。默认值：None。
        - **requires_grad** (bool)：是否需要微分求梯度。默认值：True。
        - **layerwise_parallel** (bool)：在数据/混合并行模式下， `layerwise_parallel` 配置为True时，参数广播和梯度聚合时会过滤掉该参数。默认值：False。
        - **parallel_optimizer** (bool)：用于在"semi_auto_parallel"或"auto_parallel"并行模式下区分参数是否进行优化器切分。仅在 `mindspore.context.set_auto_parallel_context()` 并行配置模块中设置 `enable_parallel_optimizer` 启用优化器并行时有效。默认值：True。

    **样例：**
    
    .. code-block::

        >>> import numpy as np
        >>> from mindspore import Parameter, Tensor
        >>> import mindspore.ops as ops
        >>> import mindspore.nn as nn
        >>> import mindspore
        >>>
        >>> class Net(nn.Cell)：
        ...     def __init__(self)：
        ...         super(Net, self).__init__()
        ...         self.matmul = ops.MatMul()
        ...         self.weight = Parameter(Tensor(np.ones((1, 2)), mindspore.float32), name="w", requires_grad=True)
        ...
        ...     def construct(self, x)：
        ...         out = self.matmul(self.weight, x)
        ...         return out
        >>> net = Net()
        >>> x = Tensor(np.ones((2, 1)), mindspore.float32)
        >>> print(net(x))
        [[2.]]
        >>> net.weight.set_data(Tensor(np.zeros((1, 2)), mindspore.float32))
        >>> print(net(x))
        [[0.]]
    

    .. py:method:: cache_enable
        :property: 

        返回该参数是否开启缓存功能。

    .. py:method::  cache_shape
        :property:

        如果开启缓存，则返回对应参数的缓存形状。

    .. py:method:: clone(init='same')

        克隆参数。

        **参数：**
                
            - **init** (Union[Tensor, str, numbers.Number])：初始化参数的形状和数据类型。如果 `init` 是 `Tensor` 或 `numbers.Number` ，则克隆一个具有相同数值、形状和数据类型的新参数。 如果 `init` 是 `str` ，则 `init` 将继承 `Initializer` 模块中对应的同名的类。例如，如果 `init` 是'same'，则克隆一个具有相同数据、形状和数据类型的新参数。默认值：'same'。

        **返回：**

            `Parameter` ，返回克隆的新参数。
        

    .. py:method:: comm_fusion

        获取和设置与此参数对应的通信算子的融合类型（int）。

        在"auto_parallel"和"semi_auto_parallel"模式下，一些用于参数或梯度聚合的通信算子将自动插入，此属性表示当前参数对应通信算子的融合类型。 `comm_fusion` 的值必须大于等于0。当 `comm_fusion` 的值为0时，算子不融合。

        仅在Ascend环境的图模式下使用。
        

    .. py:method:: data
        :property:

        返回参数对象。

    .. py:method:: init_data(layout=None, set_sliced=False)

        初始化参数的数据。

        **参数：**

            - **layout** (Union[None, tuple(list(int))])：参数切片。
                
            - **layout** [dev_mat, tensor_map, slice_shape]：默认值：None。

                - **dev_mat** (list(int))：设备矩阵。
                - **tensor_map** (list(int))：张量映射。
                - **slice_shape** (list(int))：切片形状。

            - **set_sliced** (bool)：参数初始化时被设定为分片，则为True。默认值：False。

        **异常：**

            **RuntimeError：** 参数使用 `Initializer` 模块进行初始化，初始化后并行模式发生更改。

            **ValueError：** `layout` 长度小于3。
            
            **TypeError：** `layout` 不是元组。

        **返回：**

            初始化数据后的 `Parameter` 。如果当前 `Parameter` 已初始化，则更新 `Parameter` 数据。
        

    .. py:method:: is_init
        :property:

        获取参数的初始化状态。

        此属性仅在GE（Graph Engine）中有效，在其他后端将设为False。
        

    .. py:method:: layerwise_parallel
        :property:

        在"data_parallel"或"hybrid_parallel"并行模式下，如果"layerwise_parallel"为True，参数广播和梯度聚合将不会应用到参数。
        

    .. py:method:: name
        :property:

        获取参数的名称。

    .. py:method:: parallel_optimizer
        :property:

        用于在"semi_auto_parallel"或"auto_parallel"并行模式下区分参数是否进行优化器切分。仅在 `mindspore.context.set_auto_parallel_context()` 并行配置模块中设置 `enable_parallel_optimizer` 启用优化器并行时有效。默认值：True。
        

    .. py:method:: parallel_optimizer_comm_recompute
        :property:

        在优化器并行场景下，是否重新计算与此参数对应的通信算子。

        在"auto_parallel"和"semi_auto_parallel"模式下使用优化器并行时，若参数被切分分布在不同卡上，框架会自动插入 :class:`mindspore.ops.AllGather` 通信算子用于参数汇聚。该接口用于控制 :class:`mindspore.ops.AllGather` 算子的二次计算属性。

        .. note::

            仅支持Ascend下的Graph模式。

            优化器并行场景下生成的 :class:`mindspore.ops.AllGather` 算子，建议使用 `cell.recompute(parallel_optimizer_comm_recompute=True/False)` 接口配置，不推荐直接使用本接口。
        

    .. py:method:: requires_grad
        :property:

        表示该参数是否需要求梯度进行更新。

    .. py:method:: set_data(data, slice_shape=False)

        设置参数数据。

        **参数：**

            - **data** (Union[Tensor, int, float])：新数据。
            - **slice_shape** (bool)：如果`slice_shape`设为True，则不检查 `data` 和当前参数shape的一致性。默认值：False。

        **返回：**
    
            完成数据设置的新参数。
        

    .. py:method:: set_param_fl(push_to_server=False, pull_from_server=False, requires_aggr=True)

        设置参数和服务器的交互方式。

        **参数：**

            - **push_to_server** (bool)：表示是否将参数推送到服务器。默认值：False。
            - **pull_from_server** (bool)：表示是否应从服务器中拉取参数。默认值：False。
            - **requires_aggr** (bool)：表示是否应在服务器中聚合参数。默认值：True。
        

    .. py:method:: set_param_ps(init_in_server=False)

        在Parameter Server模式下，表示训练参数是否在Server端初始化，以及是否由Server更新。

        .. note:: 仅在Parameter Server模式下有效。

        **参数：**

            - **init_in_server** (bool)：表示训练参数初始化位置是否为Server端，以及是否通过Server进行更新。默认值：False。
        

    .. py:method:: sliced
        :property:

        获取参数的切片状态。


    .. py:method:: unique
        :property:
        
        表示参数是否唯一。