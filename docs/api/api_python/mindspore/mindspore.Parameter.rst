mindspore.Parameter
========================

.. py:class:: mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True)

    `Parameter` 是 `Tensor` 的子类，当它们被绑定为Cell的属性时，会自动添加到其参数列表中，并且可以通过Cell的某些方法获取，例如 `cell.get_parameters()` 。

    .. note::
        - 在 `SEMI_AUTO_PARALLEL` 和 `AUTO_PARALLEL` 的并行模式下，如果使用 `Initializer` 模块初始化参数，参数的类型将为 `Tensor` 。`Tensor` 仅保存张量的形状和类型信息，而不占用内存来保存实际数据。
        - 并行场景下存在参数的形状发生变化的情况，用户可以调用 `Parameter` 的 `init_data` 方法得到原始数据。
        - 如果网络中存在需要部分输入为 `Parameter` 的算子，则不允许这部分输入的 `Parameter` 进行转换。

    参数：
        - **default_input** (Union[Tensor, int, float, numpy.ndarray, list]) - 初始化参数的输入值。
        - **name** (str) - 参数的名称。默认值：None。如果一个网络中存在两个及以上相同名称的 `Parameter` 对象，在定义时将提示设置一个特有的名称。
        - **requires_grad** (bool) - 是否需要微分求梯度。默认值：True。
        - **layerwise_parallel** (bool) - 在数据/混合并行模式下，`layerwise_parallel` 配置为True时，参数广播和梯度聚合时会过滤掉该参数。默认值：False。
        - **parallel_optimizer** (bool) - 用于在 `semi_auto_parallel` 或 `auto_parallel` 并行模式下区分参数是否进行优化器切分。仅在 `mindspore.context.set_auto_parallel_context()` 并行配置模块中设置 `enable_parallel_optimizer` 启用优化器并行时有效。默认值：True。

    .. py:method:: cache_enable
        :property:

        表示该参数是否开启缓存功能。

    .. py:method:: cache_shape
        :property:

        如果使用缓存，则返回对应参数的缓存shape。

    .. py:method:: clone(init='same')

        克隆参数。

        参数：
            - **init** (Union[Tensor, str, numbers.Number]) - 初始化参数的形状和数据类型。如果 `init` 是 `Tensor` 或 `numbers.Number` ，则克隆一个具有相同数值、形状和数据类型的新参数。如果 `init` 是 `str` ，则 `init` 将继承 `Initializer` 模块中对应的同名的类。例如，如果 `init` 是'same'，则克隆一个具有相同数据、形状和数据类型的新参数。默认值：'same'。

        返回：
            Parameter，返回克隆的新参数。

    .. py:method:: comm_fusion
        :property:

        获取此参数的通信算子的融合类型（int）。

        在 `AUTO_PARALLEL` 和 `SEMI_AUTO_PARALLEL` 模式下，一些用于参数或梯度聚合的通信算子将自动插入。fusion的值必须大于等于0。当fusion的值为0时，算子不会融合在一起。

    .. py:method:: data
        :property:

        返回参数对象。

    .. py:method:: init_data(layout=None, set_sliced=False)

        初始化参数的数据。

        参数：
            - **layout** (Union[None, tuple]) - 参数的layout信息。layout[dev_mat, tensor_map, slice_shape, filed_size, uniform_split, opt_shard_group]：默认值：None。仅在 `SEMI_AUTO_PARALLEL` 或 `AUTO_PARALLEL` 模式下layout不是None。

              - **dev_mat** (list(int)) - 该参数的设备矩阵。
              - **tensor_map** (list(int)) - 该参数的张量映射。
              - **slice_shape** (list(int)) - 该参数的切片shape。
              - **filed_size** (int) - 该权重的行数。
              - **uniform_split** (bool) - 该参数是否进行均匀切分。
              - **opt_shard_group** (str) - 该参数进行优化器切分时的group。

            - **set_sliced** (bool) - 参数初始化时被设定为分片，则为True。默认值：False。

        返回：
            初始化数据后的 `Parameter` 。如果当前 `Parameter` 已初始化，则更新 `Parameter` 数据。

        异常：
            - **RuntimeError** - 参数使用 `Initializer` 模块进行初始化，初始化后并行模式发生更改。
            - **ValueError** - `layout` 长度小于6。
            - **TypeError** - `layout` 不是元组。

    .. py:method:: inited_param
        :property:

        用于调用 `init_data` 后，获取当前的Parameter。

        如果 `self` 是没有数据的Parameter，则默认返回为None；在调用 `init_data` 方法对Parameter初始化数据后，当前Parameter会被记录在此属性中。

    .. py:method:: key
        :property:

        用于获取当前Parameter的唯一key值。

    .. py:method:: layerwise_parallel
        :property:

        获取此参数的逐层并行状态（bool）。

        在 `DATA_PARALLEL` 和 `HYBRID_PARALLEL` 模式下，如果 `layerwise_parallel` 为True，则广播和gradients通信将不会应用到参数。

    .. py:method:: name
        :property:

        获取参数的名称。

    .. py:method:: parallel_optimizer
        :property:

        获取此参数的优化器并行状态（bool）。

        用于在 `AUTO_PARALLEL` 和 `SEMI_AUTO_PARALLEL` 模式下过滤权重切分操作。当在 `mindspore.context.set_auto_parallel_context()` 中启用优化器并行时，它才有效。

    .. py:method:: parallel_optimizer_comm_recompute
        :property:

        获取此参数的优化器并行通信重计算状态（bool）。

        在 `AUTO_PARALLEL` 和 `SEMI_AUTO_PARALLEL` 模式下，当使用并行优化器时，会自动插入一些 :class:`mindspore.ops.AllGather` 算子，用于参数聚合。它用于控制这些 :class:`mindspore.ops.AllGather` 算子的重计算属性。

        .. note::
            - 仅支持 `Graph` 模式。
            - 建议使用cell.recompute(parallel_optimizer_comm_recompute=True/False)去配置由优化器并行生成的 :class:`mindspore.ops.AllGather` 算子，而不是直接使用该接口。

    .. py:method:: requires_grad
        :property:

        表示该参数是否需要求梯度进行更新。

    .. py:method:: set_data(data, slice_shape=False)

        设置参数数据。

        参数：
            - **data** (Union[Tensor, int, float]) - 新数据。
            - **slice_shape** (bool) - 如果 `slice_shape` 设为True，则不检查 `data` 和当前参数shape的一致性。默认值：False。

        返回：
            完成数据设置的新参数。

    .. py:method:: set_param_fl(push_to_server=False, pull_from_server=False, requires_aggr=True)

        设置参数和服务器的互动方式。

        参数：
            - **push_to_server** (bool) - 表示是否将参数推送到服务器。默认值：False。
            - **pull_from_server** (bool) - 表示是否应从服务器中拉取参数。默认值：False。
            - **requires_aggr** (bool) - 表示是否应在服务器中聚合参数。默认值：True。

    .. py:method:: set_param_ps(init_in_server=False)

        表示可训练参数是否由参数服务器更新，以及可训练参数是否在服务器上初始化。

        .. note:: 仅当运行的任务处于参数服务器模式下有效。

        参数：
            - **init_in_server** (bool) - 表示参数服务器更新的可训练参数是否在服务器上初始化。默认值：False。

    .. py:method:: sliced
        :property:

        获取参数的切片状态。

    .. py:method:: unique
        :property:

        表示参数是否唯一。
