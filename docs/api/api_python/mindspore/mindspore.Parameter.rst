mindspore.Parameter
========================

.. py:class:: mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True, storage_format="")

    `Parameter` 是 `Tensor` 的子类，当它们被绑定为Cell的属性时，会自动添加到其参数列表中，并且可以通过Cell的某些方法获取，例如 `cell.get_parameters()` 。

    .. note::
        - 在 `SEMI_AUTO_PARALLEL` 和 `AUTO_PARALLEL` 的自动并行模式下，如果使用 `Initializer` 模块初始化参数，参数的类型将为 `Tensor` 。`Tensor` 仅保存张量的形状和类型信息，而不占用内存来保存实际数据。
        - 并行场景下存在参数的形状发生变化的情况，用户可以调用 `Parameter` 的 `init_data` 方法得到原始数据。
        - 如果网络中存在需要部分输入为 `Parameter` 的算子，则不允许这部分输入的 `Parameter` 进行数据类型转换。
        - 每一个 `Parameter` 使用唯一的名字可以帮助后续的操作和更新。如果有两个或多个 `Parameter` 在同一个网络中使用了相同的名字，将会提示在定义时使用唯一的名字。
        - `Parameter` 直接打印时无法查看到里面实际包含的值，需要使用 `Parameter.asnumpy()` 方法来查看实际的值。

    参数：
        - **default_input** (Union[Tensor, int, float, numpy.ndarray, list]) - 初始化参数的输入值。
        - **name** (str) - 参数的名称。默认值： ``None`` 。如果一个网络中存在两个及以上相同名称的 `Parameter` 对象，在定义时将提示设置一个特有的名称。
          
          1. 如果一个 `Parameter` 未命名，默认的名字就是变量名。例如，`param_a` 的名字是 `name_a`，`param_b` 的名字是 `param_b` 。

          .. code-block::

              self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
              self.param_b = Parameter(Tensor([2], ms.float32))

          2. 如果在list或tuple中的 `Parameter` 未命名，将会提供一个唯一值。例如，以下 `Parameter` 的名字是 **Parameter$1** and **Parameter$2**。

          .. code-block::

              self.param_list = [Parameter(Tensor([3], ms.float32)),
                                 Parameter(Tensor([4], ms.float32))]

          3. 如果 `Parameter` 已命名， 并且不同 `Parameter` 间有重复名称，将会抛出异常。例如，"its name 'name_a' already exists."将会抛出。

          .. code-block::

              self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
              self.param_tuple = (Parameter(Tensor([5], ms.float32), name="name_a"),
                                  Parameter(Tensor([6], ms.float32)))

          4. 如果一个 `Parameter` 多次出现在list或tuple中，只检查一次他的名字。例如，以下代码将不会抛出异常。

          .. code-block::

              self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
              self.param_tuple = (self.param_a, self.param_a)

        - **requires_grad** (bool) - 是否需要微分求梯度。默认值： ``True`` 。
        - **layerwise_parallel** (bool) - 在数据/混合并行模式下， `layerwise_parallel` 配置为 ``True`` 时，参数广播和梯度聚合时会过滤掉该 `Parameter` 。默认值： ``False`` 。
        - **parallel_optimizer** (bool) - 用于在 `SEMI_AUTO_PARALLEL` 或 `AUTO_PARALLEL` 并行模式下区分该参数是否进行优化器切分。仅在 `mindspore.set_auto_parallel_context()` 并行配置模块中设置 `enable_parallel_optimizer` 启用优化器并行时有效。默认值： ``True`` 。
        - **storage_format** (str) - 仅限Ascend，用于指定权重加载到设备的格式。默认不改变格式，可选值为： ``"FRACTAL_NZ"`` 、 ``"NC1HWC0"`` 、 ``"FRACTAL_Z"`` 等。默认值： ``""`` 。

    .. py:method:: add_pipeline_stage(stage)

        为参数添加pipeline_stage。

        参数：
            - **stage** (int) - 参数的pipeline_stage。

        异常：
            - **TypeError** - 如果 `stage` 不是正整数。

    .. py:method:: cache_enable
        :property:

        表示该参数是否开启缓存功能。

    .. py:method:: cache_shape
        :property:

        如果使用缓存，则返回对应参数的缓存shape。

    .. py:method:: clone(init='same')

        克隆参数。

        参数：
            - **init** (Union[Tensor, str, numbers.Number]) - 初始化参数的形状和数据类型。如果 `init` 是 `Tensor` 或 `numbers.Number` ，则克隆一个具有相同数值、形状和数据类型的新参数。如果 `init` 是 `str` ，则 `init` 将继承 `Initializer` 模块中对应的同名的类。例如，如果 `init` 是 ``'same'`` ，则克隆一个具有相同数据、形状和数据类型的新参数。默认值： ``'same'`` 。

        返回：
            Parameter，返回克隆的新参数。

    .. py:method:: comm_fusion
        :property:

        获取此参数的通信算子的融合类型（int）。

        在 `AUTO_PARALLEL` 和 `SEMI_AUTO_PARALLEL` 模式下，一些用于参数或梯度聚合的通信算子将自动插入。 `comm_fusion` 的值必须大于等于0。当 `comm_fusion` 为 ``0`` 时，算子不会融合在一起。

    .. py:method:: copy

        拷贝参数。

        返回：
            Parameter，返回拷贝的新参数。

    .. py:method:: data
        :property:

        返回参数对象。

    .. py:method:: init_data(layout=None, set_sliced=False)

        初始化参数的数据。

        参数：
            - **layout** (Union[None, tuple]) - 参数的layout信息。layout[dev_mat, tensor_map, slice_shape, filed_size, uniform_split, opt_shard_group]。默认值： ``None`` 。仅在 `SEMI_AUTO_PARALLEL` 或 `AUTO_PARALLEL` 模式下 `layout` 不是 ``None`` 。

              - **dev_mat** (list(int)) - 该参数的设备矩阵。
              - **tensor_map** (list(int)) - 该参数的张量映射。
              - **slice_shape** (list(int)) - 该参数的切片shape。
              - **filed_size** (int) - 该权重的行数。
              - **uniform_split** (bool) - 该参数是否进行均匀切分。
              - **opt_shard_group** (str) - 该参数进行优化器切分时的group。

            - **set_sliced** (bool) - 参数初始化时被设定为分片，则为 ``True`` 。默认值： ``False`` 。

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

        在 `DATA_PARALLEL` 和 `HYBRID_PARALLEL` 模式下，如果 `layerwise_parallel` 为 ``True`` ，则广播和gradients通信将不会应用到参数。

    .. py:method:: name
        :property:

        获取参数的名称。

    .. py:method:: parallel_optimizer
        :property:

        获取此参数的优化器并行状态（bool）。

        用于在 `AUTO_PARALLEL` 和 `SEMI_AUTO_PARALLEL` 模式下过滤权重切分操作。当在 `mindspore.set_auto_parallel_context()` 中启用优化器并行时，它才有效。

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
            - **slice_shape** (bool) - 如果 `slice_shape` 设为 ``True`` ，则不检查 `data` 和当前参数shape的一致性。默认值： ``False`` 。当 `slice_shape` 设为 ``True`` 时，如果两个shape不一致，会抛出ValueError。

        返回：
            完成数据设置的新参数。

    .. py:method:: set_param_ps(init_in_server=False)

        表示可训练参数是否由参数服务器更新，以及可训练参数是否在服务器上初始化。

        .. note:: 
            仅当运行的任务处于参数服务器模式下有效。
            只支持在图模式下调用。

        参数：
            - **init_in_server** (bool) - 表示参数服务器更新的可训练参数是否在服务器上初始化。默认值： ``False`` 。

        教程样例：
            - `Parameter Server模式
              <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html>`_

    .. py:method:: sliced
        :property:

        获取参数的切片状态。

    .. py:method:: unique
        :property:

        表示参数是否唯一。

    .. py:method:: value()

        返回参数的值。