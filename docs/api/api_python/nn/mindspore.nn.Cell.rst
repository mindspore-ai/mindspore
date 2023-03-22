mindspore.nn.Cell
==================

.. py:class:: mindspore.nn.Cell(auto_prefix=True, flags=None)

    MindSpore中神经网络的基本构成单元。模型或神经网络层应当继承该基类。

    `mindspore.nn` 中神经网络层也是Cell的子类，如 :class:`mindspore.nn.Conv2d` 、 :class:`mindspore.nn.ReLU` 等。Cell在GRAPH_MODE(静态图模式)下将编译为一张计算图，在PYNATIVE_MODE(动态图模式)下作为神经网络的基础模块。

    参数：
        - **auto_prefix** (bool，可选) - 是否自动为Cell及其子Cell生成NameSpace。`auto_prefix` 的设置影响网络参数的命名，如果设置为True，则自动给网络参数的名称添加前缀，否则不添加前缀。默认值：True。
        - **flags** (dict，可选) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。

    .. py:method:: add_flags(**flags)

        为Cell添加自定义属性。

        在实例化Cell类时，如果入参flags不为空，会调用此方法。

        参数：
            - **flags** (dict) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。

    .. py:method:: add_flags_recursive(**flags)

        如果Cell含有多个子Cell，此方法会递归得给所有子Cell添加自定义属性。

        参数：
            - **flags** (dict) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。

    .. py:method:: apply(fn)

        递归地将 `fn` 应用于每个子Cell（由 `.cells()` 返回）以及自身。通常用于初始化模型的参数。

        参数：
            - **fn** (function) - 被执行于每个Cell的function。
        
        返回：
            Cell类型，Cell本身。

    .. py:method:: auto_cast_inputs(inputs)

        在混合精度下，自动对输入进行类型转换。

        参数：
            - **inputs** (tuple) - construct方法的输入。

        返回：
            Tuple类型，经过类型转换后的输入。

    .. py:method:: bprop_debug
        :property:

        在图模式下使用，用于标识是否使用自定义的反向传播函数。

    .. py:method:: cast_inputs(inputs, dst_type)

        将输入转换为指定类型。

        参数：
            - **inputs** (tuple[Tensor]) - 输入。
            - **dst_type** (mindspore.dtype) - 指定的数据类型。

        返回：
            tuple[Tensor]类型，转换类型后的结果。

    .. py:method:: cast_param(param)

        在PyNative模式下，根据自动混合精度的精度设置转换Cell中参数的类型。

        该接口目前在自动混合精度场景下使用。

        参数：
            - **param** (Parameter) - 需要被转换类型的输入参数。

        返回：
            Parameter类型，转换类型后的参数。

    .. py:method:: cells()

        返回当前Cell的子Cell的迭代器。

        返回：
            Iteration类型，Cell的子Cell。

    .. py:method:: cells_and_names(cells=None, name_prefix='')

        递归地获取当前Cell及输入 `cells` 的所有子Cell的迭代器，包括Cell的名称及其本身。

        参数：
            - **cells** (str) - 需要进行迭代的Cell。默认值：None。
            - **name_prefix** (str) - 作用域。默认值： ''。

        返回：
            Iteration类型，当前Cell及输入 `cells` 的所有子Cell和相对应的名称。

    .. py:method:: check_names()

        检查Cell中的网络参数名称是否重复。

    .. py:method:: compile(*args, **kwargs)

        编译Cell为计算图，输入需与construct中定义的输入一致。

        参数：
            - **args** (tuple) - Cell的输入。
            - **kwargs** (dict) - Cell的输入。

    .. py:method:: compile_and_run(*args, **kwargs)

        编译并运行Cell，输入需与construct中定义的输入一致。

        .. note::
            不推荐使用该函数，建议直接调用Cell实例。

        参数：
            - **args** (tuple) - Cell的输入。
            - **kwargs** (dict) - Cell的输入。

        返回：
            Object类型，执行的结果。

    .. py:method:: construct(*args, **kwargs)

        定义要执行的计算逻辑。所有子类都必须重写此方法。

        .. note::
            当前不支持inputs同时输入tuple类型和非tuple类型。

        参数：
            - **args** (tuple) - 可变参数列表，默认值：()。
            - **kwargs** (dict) - 可变的关键字参数的字典，默认值：{}。

        返回：
            Tensor类型，返回计算结果。

    .. py:method:: exec_checkpoint_graph()

        保存checkpoint图。

    .. py:method:: extend_repr()

        在原有描述基础上扩展Cell的描述。

        若需要在print时输出个性化的扩展信息，请在您的网络中重新实现此方法。

    .. py:method:: flatten_weights(fusion_size=0)

        重置权重参数（即可训练参数）使用的数据内存，让这些参数按数据类型分组使用连续内存块。

        .. note::
            默认情况下，具有相同数据类型的参数会使用同一个连续内存块。但对于某些具有大量参数的模型，
            将一个大的连续内存块分为多个小一点的内存块有可能提升性能，对于这种情况，
            可以通过 `fusion_size` 参数来限制最大连续内存块的的大小。

        参数：
            - **fusion_size** (int) - 最大连续内存块的大小（以字节为单位），0表示不限制大小。默认值：0。

    .. py:method:: generate_scope()

        为网络中的每个Cell对象生成NameSpace。

    .. py:method:: get_flags()

        获取该Cell的自定义属性，自定义属性通过 `add_flags` 方法添加。

    .. py:method:: get_func_graph_proto()

        返回图的二进制原型。

    .. py:method:: get_inputs()

        返回编译计算图所设置的输入。

        返回：
            Tuple类型，编译计算图所设置的输入。

        .. note::
            这是一个实验接口，可能会被更改或者删除。

    .. py:method:: get_parameters(expand=True)

        返回Cell中parameter的迭代器。

        获取Cell的参数。如果 `expand` 为true，获取此cell和所有subcells的参数。

        参数：
            - **expand** (bool) - 如果为True，则递归地获取当前Cell和所有子Cell的parameter。否则，只生成当前Cell的子Cell的parameter。默认值：True。

        返回：
            Iteration类型，Cell的parameter。

    .. py:method:: get_scope()

        返回Cell的作用域。

        返回：
            String类型，网络的作用域。

    .. py:method:: infer_param_pipeline_stage()

        推导Cell中当前 `pipeline_stage` 的参数。

        .. note::
            - 如果某参数不属于任何已被设置 `pipeline_stage` 的Cell，此参数应使用 `add_pipeline_stage` 方法来添加它的 `pipeline_stage` 信息。
            - 如果某参数P被stageA和stageB两个不同stage的算子使用，那么参数P在使用 `infer_param_pipeline_stage` 之前，应使用 `P.add_pipeline_stage(stageA)` 和 `P.add_pipeline_stage(stageB)` 添加它的stage信息。

        返回：
            属于当前 `pipeline_stage` 的参数。

        异常：
            - **RuntimeError** - 如果参数不属于任何stage。

    .. py:method:: init_parameters_data(auto_parallel_mode=False)

        初始化并替换Cell中所有的parameter的值。

        .. note::
            在调用 `init_parameters_data` 后，`trainable_params()` 或其他相似的接口可能返回不同的参数对象，不要保存这些结果。

        参数：
            - **auto_parallel_mode** (bool) - 是否在自动并行模式下执行。默认值：False。

        返回：
            Dict[Parameter, Parameter]，返回一个原始参数和替换参数的字典。

    .. py:method:: insert_child_to_cell(child_name, child_cell)

        将一个给定名称的子Cell添加到当前Cell。

        参数：
            - **child_name** (str) - 子Cell名称。
            - **child_cell** (Cell) - 要插入的子Cell。

        异常：
            - **KeyError** - 如果子Cell的名称不正确或与其他子Cell名称重复。
            - **TypeError** - 如果 `child_name` 的类型不为str类型。
            - **TypeError** - 如果子Cell的类型不正确。

    .. py:method:: insert_param_to_cell(param_name, param, check_name_contain_dot=True)

        向当前Cell添加参数。

        将指定名称的参数添加到Cell中。目前在 `mindspore.nn.Cell.__setattr__` 中使用。

        参数：
            - **param_name** (str) - 参数名称。
            - **param** (Parameter) - 要插入到Cell的参数。
            - **check_name_contain_dot** (bool) - 是否对 `param_name` 中的"."进行检查。默认值：True。

        异常：
            - **KeyError** - 如果参数名称为空或包含"."。
            - **TypeError** - 如果参数的类型不是Parameter。

    .. py:method:: name_cells()

        递归地获取一个Cell中所有子Cell的迭代器。

        包括Cell名称和Cell本身。

        返回：
            Dict[String, Cell]，Cell中的所有子Cell及其名称。

    .. py:method:: param_prefix
        :property:

        当前Cell的子Cell的参数名前缀。

    .. py:method:: parameter_layout_dict
        :property:

        `parameter_layout_dict` 表示一个参数的张量layout，这种张量layout是由分片策略和分布式算子信息推断出来的。

    .. py:method:: parameters_and_names(name_prefix='', expand=True)

        返回Cell中parameter的迭代器。

        包含参数名称和参数本身。

        参数：
            - **name_prefix** (str) - 作用域。默认值： ''。
            - **expand** (bool) - 如果为True，则递归地获取当前Cell和所有子Cell的参数及名称；如果为False，只生成当前Cell的子Cell的参数及名称。默认值：True。

        返回：
            迭代器，Cell的名称和Cell本身。

    .. py:method:: parameters_broadcast_dict(recurse=True)

        获取这个Cell的参数广播字典。

        参数：
            - **recurse** (bool) - 是否包含子Cell的参数。默认值：True。

        返回：
            OrderedDict，返回参数广播字典。

    .. py:method:: parameters_dict(recurse=True)

        获取此Cell的parameter字典。

        参数：
            - **recurse** (bool) - 是否递归得包含所有子Cell的parameter。默认值：True。

        返回：
            OrderedDict类型，返回参数字典。

    .. py:method:: place(role, rank_id)
        
        为该Cell中所有算子设置标签。此标签告诉MindSpore编译器此Cell在哪个进程上启动。
        每个标签都由进程角色 `role` 和 `rank_id` 组成，因此，通过对不同Cell设置不同标签，这些Cell将在不同进程启动，使用户可以进行分布式训练/推理等任务。

        .. note::
            - 此接口只在成功调用 `mindspore.communication.init()` 完成动态组网后才能生效。

        参数：
            - **role** (str) - 算子执行所在进程的角色。只支持'MS_WORKER'。
            - **rank_id** (int) - 算子执行所在进程的id。在相同进程角色间， `rank_id` 是唯一的。

    .. py:method:: recompute(**kwargs)

        设置Cell重计算。Cell中输出算子以外的所有算子将被设置为重计算。如果一个算子的计算结果被输出到一些反向节点来进行梯度计算，且被设置成重计算，那么我们会在反向传播中重新计算它，而不去存储在前向传播中的中间激活层的计算结果。

        .. note::
            - 如果计算涉及到诸如随机化或全局变量之类的操作，那么目前还不能保证等价。
            - 如果该Cell中算子的重计算API也被调用，则该算子的重计算模式以算子的重计算API的设置为准。
            - 该接口仅配置一次，即当父Cell配置了，子Cell不需再配置。
            - Cell的输出算子默认不做重计算，这一点是基于我们减少内存占用的配置经验。如果一个Cell里面只有一个算子而且想要把这个算子设置为重计算的，那么请使用算子的重计算API。
            - 当应用了重计算且内存充足时，可以配置'mp_comm_recompute=False'来提升性能。
            - 当应用了重计算但内存不足时，可以配置'parallel_optimizer_comm_recompute=True'来节省内存。有相同融合group的Cell应该配置相同的parallel_optimizer_comm_recompute。

        参数：
            - **mp_comm_recompute** (bool) - 表示在自动并行或半自动并行模式下，指定Cell内部由模型并行引入的通信操作是否重计算。默认值：True。
            - **parallel_optimizer_comm_recompute** (bool) - 表示在自动并行或半自动并行模式下，指定Cell内部由优化器并行引入的AllGather通信是否重计算。默认值：False。

    .. py:method:: register_backward_hook(hook_fn)

        设置Cell对象的反向hook函数。

        .. note::
            - `register_backward_hook(hook_fn)` 在图模式下，或者在PyNative模式下使用 `jit` 装饰器功能时不起作用。
            - hook_fn必须有如下代码定义。 `cell_id` 是已注册Cell对象的信息，包括名称和ID。 `grad_input` 是反向传递给Cell对象的梯度。 `grad_output` 是Cell对象的反向输出梯度。用户可以在hook_fn中打印梯度数据或者返回新的输出梯度。
            - hook_fn返回新的输出梯度或者None：hook_fn(cell_id, grad_input, grad_output) -> New grad_output or None。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用 `register_backward_hook(hook_fn)` 。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用 `register_backward_hook(hook_fn)` ，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和反向输入，输出梯度的hook_fn函数。

        返回：
            `mindspore.common.hook_handle.HookHandle` 类型，与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_forward_hook(hook_fn)

        设置Cell对象的正向hook函数。

        .. note::
            - `register_forward_hook(hook_fn)` 在图模式下，或者在PyNative模式下使用 `jit` 装饰器功能时不起作用。
            - hook_fn必须有如下代码定义。 `cell_id` 是已注册Cell对象的信息，包括名称和ID。 `inputs` 是网络正向传播时Cell对象的输入数据。 `outputs` 是网络正向传播时Cell对象的输出数据。用户可以在hook_fn中打印数据或者返回新的输出数据。
            - hook_fn返回新的输出数据或者None：hook_fn(cell_id, inputs, outputs) -> New outputs or None。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用 `register_forward_hook(hook_fn)` 。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用 `register_forward_hook(hook_fn)` ，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和正向输入，输出数据的hook_fn函数。

        返回：
            `mindspore.common.hook_handle.HookHandle` 类型，与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_forward_pre_hook(hook_fn)

        设置Cell对象的正向pre_hook函数。

        .. note::
            - `register_forward_pre_hook(hook_fn)` 在图模式下，或者在PyNative模式下使用 `jit` 装饰器功能时不起作用。
            - hook_fn必须有如下代码定义。 `cell_id` 是已注册Cell对象的信息，包括名称和ID。 `inputs` 是网络正向传播时Cell对象的输入数据。用户可以在hook_fn中打印输入数据或者返回新的输入数据。
            - hook_fn返回新的输入数据或者None：hook_fn(cell_id, inputs) -> New inputs or None。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用 `register_forward_pre_hook(hook_fn)` 。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用 `register_forward_pre_hook(hook_fn)` ，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和正向输入数据的hook_fn函数。

        返回：
            `mindspore.common.hook_handle.HookHandle` 类型，与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: remove_redundant_parameters()

        删除冗余参数。

        这个接口通常不需要显式调用。

    .. py:method:: run_construct(cast_inputs, kwargs)

        运行construct方法。

        .. note::
            该函数已经弃用，将会在未来版本中删除。不推荐使用此函数。

        参数：
            - **cast_inputs** (tuple) - Cell的输入。
            - **kwargs** (dict) - 关键字参数。

        返回：
            Cell的输出。

    .. py:method:: set_boost(boost_type)

        为了提升网络性能，可以配置boost内的算法让框架自动使能该算法来加速网络训练。

        请确保 `boost_type` 所选择的算法在
        `algorithm library <https://gitee.com/mindspore/mindspore/tree/master/mindspore/python/mindspore/boost>`_ 算法库中。

        .. note:: 部分加速算法可能影响网络精度，请谨慎选择。

        参数：
            - **boost_type** (str) - 加速算法。

        返回：
            Cell类型，Cell本身。

        异常：
            - **ValueError** - 如果 `boost_type` 不在boost算法库内。

    .. py:method:: set_broadcast_flag(mode=True)

        设置该Cell的参数广播模式。

        参数：
            - **mode** (bool) - 指定当前模式是否进行参数广播。默认值：True。

    .. py:method:: set_comm_fusion(fusion_type, recurse=True)

        为Cell中的参数设置融合类型。请参考 :class:`mindspore.Parameter.comm_fusion` 的描述。

        .. note:: 当函数被多次调用时，此属性值将被重写。

        参数：
            - **fusion_type** (int) - Parameter的 `comm_fusion` 属性的设置值。
            - **recurse** (bool) - 是否递归地设置子Cell的可训练参数。默认值：True。

    .. py:method:: set_data_parallel()

        递归设置该Cell中的所有算子的并行策略为数据并行。

        .. note:: 仅在图模式，使用auto_parallel_context = ParallelMode.AUTO_PARALLEL生效。

    .. py:method:: set_grad(requires_grad=True)

        Cell的梯度设置。在PyNative模式下，该参数指定Cell是否需要梯度。如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。

        参数：
            - **requires_grad** (bool) - 指定网络是否需要梯度，如果为True，PyNative模式下Cell将构建反向网络。默认值：True。

        返回：
            Cell类型，Cell本身。

    .. py:method:: set_inputs(*inputs)

        设置编译计算图所需的输入。输入数量需与数据集数量一致。若使用Model接口，请确保所有传入Model的网络和损失函数都配置了set_inputs。
        输入可以为动态或静态的Tensor。

        参数：
            - **inputs** (tuple) - Cell的输入。

        .. note::
            这是一个实验接口，可能会被更改或者删除。

    .. py:method:: set_jit_config(jit_config)

        为Cell设置编译时所使用的JitConfig配置项。

        参数：
            - **jit_config** (JitConfig) - Cell的Jit配置信息。目前支持下面两个配置项。
  
              - **jit_level** (str) - 用于设置优化图的'level'参数。取值范围['O0'、'O1'、'O2']。默认值：'O1'。

                - O0：基本优化。
                - O1：手动优化。
                - O2：手动优化和图算融合。

              - **task_sink** (bool) - 是否通过数据集方式传递数据。默认值：True。

    .. py:method:: set_param_fl(push_to_server=False, pull_from_server=False, requires_aggr=True)

        设置参数与服务器交互的方式。

        参数：
            - **push_to_server** (bool) - 是否将参数推送到服务器。默认值：False。
            - **pull_from_server** (bool) - 是否从服务器提取参数。默认值：False。
            - **requires_aggr** (bool) - 是否在服务器中聚合参数。默认值：True。

    .. py:method:: set_param_ps(recurse=True, init_in_server=False)

        设置可训练参数是否由参数服务器更新，以及是否在服务器上初始化可训练参数。

        .. note::
            只在运行的任务处于参数服务器模式时有效。
            只支持在图模式下调用。

        参数：
            - **recurse** (bool) - 是否设置子网络的可训练参数。默认值：True。
            - **init_in_server** (bool) - 是否在服务器上初始化由参数服务器更新的可训练参数。默认值：False。

    .. py:method:: set_train(mode=True)

        将Cell设置为训练模式。

        设置当前Cell和所有子Cell的训练模式。对于训练和预测具有不同结构的网络层(如 `BatchNorm`)，将通过这个属性区分分支。如果设置为True，则执行训练分支，否则执行另一个分支。

        .. note::
            当执行Model.train()的时候，框架会默认调用Cell.set_train(True)。
            当执行Model.eval()的时候，框架会默认调用Cell.set_train(False)。

        参数：
            - **mode** (bool) - 指定模型是否为训练模式。默认值：True。

        返回：
            Cell类型，Cell本身。

    .. py:method:: shard(in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0)

        指定输入/输出Tensor的分布策略，其余算子的策略推导得到。在PyNative模式下，可以利用此方法指定某个Cell以图模式进行分布式执行。 in_strategy/out_strategy需要为元组类型，
        其中的每一个元素指定对应的输入/输出的Tensor分布策略，可参考： `mindspore.ops.Primitive.shard` 的描述。也可以设置为None，会默认以数据并行执行。
        其余算子的并行策略由输入输出指定的策略推导得到。

        .. note:: 需设置为PyNative模式，并且ParallelMode.AUTO_PARALLEL，
            同时设置 `set_auto_parallel_context` 中的搜索模式(search mode)为"sharding_propagation"。
            如果输入含有Parameter，其对应的策略应该在 `in_strategy` 里设置。

        参数：
            - **in_strategy** (tuple) - 指定各输入的切分策略，输入元组的每个元素可以为元组或None，元组即具体指定输入每一维的切分策略，None则会默认以数据并行执行。
            - **out_strategy** (Union[None, tuple]) - 指定各输出的切分策略，用法同in_strategy，目前未使能。默认值：None。
            - **parameter_plan** (Union[dict, None]) - 指定各参数的切分策略，传入字典时，键是str类型的参数名，值是一维整数tuple表示相应的切分策略，
              如果参数名错误或对应参数已经设置了切分策略，该参数的设置会被跳过。默认值：None。
            - **device** (string) - 指定执行设备，可以为["CPU", "GPU", "Ascend"]中任意一个，目前未使能。默认值："Ascend"。
            - **level** (int) - 指定搜索切分策略的目标函数，即是最大化计算通信比、最小化内存消耗、最大化执行速度等。可以为[0, 1, 2]中任意一个，默认值：0。目前仅支持最大化计算通信比，其余模式未使能。

        返回：
            Cell类型，Cell本身。

    .. py:method:: to_float(dst_type)

        在Cell和所有子Cell的输入上添加类型转换，以使用特定的浮点类型运行。

        如果 `dst_type` 是 `mindspore.dtype.float16` ，Cell的所有输入(包括作为常量的input， Parameter， Tensor)都会被转换为float16。请参考 :func:`mindspore.amp.build_train_network` 的源代码中的用法。

        .. note:: 多次调用将产生覆盖。

        参数：
            - **dst_type** (mindspore.dtype) - Cell转换为 `dst_type` 类型运行。 `dst_type` 可以是 `mindspore.dtype.float16` 或者  `mindspore.dtype.float32` 。

        返回：
            Cell类型，Cell本身。

        异常：
            - **ValueError** - 如果 `dst_type` 不是 `mindspore.dtype.float32` ，也不是 `mindspore.dtype.float16`。

    .. py:method:: trainable_params(recurse=True)

        返回Cell的可训练参数。

        返回一个可训练参数的列表。

        参数：
            - **recurse** (bool) - 是否递归地包含当前Cell的所有子Cell的可训练参数。默认值：True。

        返回：
            List类型，可训练参数列表。

    .. py:method:: untrainable_params(recurse=True)

        返回Cell的不可训练参数。

        返回一个不可训练参数的列表。

        参数：
            - **recurse** (bool) - 是否递归地包含当前Cell的所有子Cell的不可训练参数。默认值：True。

        返回：
            List类型，不可训练参数列表。

    .. py:method:: update_cell_prefix()

        递归地更新所有子Cell的 `param_prefix` 。

        在调用此方法后，可以通过Cell的 `param_prefix` 属性获取该Cell的所有子Cell的名称前缀。

    .. py:method:: update_cell_type(cell_type)

        量化感知训练网络场景下，更新当前Cell的类型。

        此方法将Cell类型设置为 `cell_type` 。

        参数：
            - **cell_type** (str) - 被更新的类型，`cell_type` 可以是"quant"或"second-order"。

    .. py:method:: update_parameters_name(prefix='', recurse=True)

        给网络参数名称添加 `prefix` 前缀字符串。

        参数：
            - **prefix** (str) - 前缀字符串。默认值： ''。
            - **recurse** (bool) - 是否递归地包含所有子Cell的参数。默认值：True。
