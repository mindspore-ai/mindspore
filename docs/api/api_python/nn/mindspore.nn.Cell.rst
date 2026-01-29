mindspore.nn.Cell
==================

.. py:class:: mindspore.nn.Cell(auto_prefix=True, flags=None)

    MindSpore中神经网络的基本构成单元。模型或神经网络层应当继承该基类。

    `mindspore.nn` 中神经网络层也是Cell的子类，如 :class:`mindspore.nn.Conv2d` 、 :class:`mindspore.nn.ReLU` 等。Cell在GRAPH_MODE（静态图模式）下将编译为一张计算图，在PYNATIVE_MODE（动态图模式）下作为神经网络的基础模块。

    .. note::
        Cell默认情况下是推理模式。对于继承Cell的类，如果训练和推理具有不同结构，子类会默认执行推理分支。设置训练模式，请参考 `mindspore.nn.Cell.set_train` 。

    .. warning::
        在Cell的子类中不能定义名为'cast'的方法，不能定义名为'phase'和'cells'的属性, 否则会报错。

    参数：
        - **auto_prefix** (bool，可选) - 是否自动为Cell及其子Cell生成NameSpace。该参数同时会影响 `Cell` 中权重参数的名称。如果设置为 ``True`` ，则自动给权重参数的名称添加前缀，否则不添加前缀。通常情况下，骨干网络应设置为 ``True`` ，否则会产生重名问题。用于训练骨干网络的优化器、 :class:`mindspore.nn.TrainOneStepCell` 等，应设置为 ``False`` ，否则骨干网络的权重参数名会被误改。默认值： ``True`` 。
        - **flags** (dict，可选) - Cell的配置信息，目前用于绑定Cell和数据集。用户也可通过该参数自定义Cell属性。默认值： ``None`` 。

    .. py:method:: add_flags(**flags)

        为Cell添加自定义属性。

        在实例化Cell类时，如果入参flags不为空，会调用此方法。

        参数：
            - **flags** (dict) - Cell的配置信息，目前用于绑定Cell和数据集。用户也可通过该参数自定义Cell属性。

    .. py:method:: add_flags_recursive(**flags)

        如果Cell含有多个子Cell，此方法会递归地给所有子Cell添加自定义属性。

        参数：
            - **flags** (dict) - Cell的配置信息，目前用于绑定Cell和数据集。用户也可通过该参数自定义Cell属性。

    .. py:method:: apply(fn)

        递归地将 `fn` 应用于每个子Cell（由 `.cells()` 返回）以及自身。通常用于初始化模型的参数。

        参数：
            - **fn** (function) - 被执行于每个Cell的function。

        返回：
            Cell类型，Cell本身。

    .. py:method:: bprop_debug
        :property:

        在图模式下使用，用于标识是否使用自定义的反向传播函数。


    .. py:method:: buffers(recurse=True)

        返回Cell缓冲区的迭代器，只包含缓冲区本身。

        参数：
            - **recurse** (bool，可选) - 如果为 ``True`` ，则生成此Cell及其子Cell的缓冲区。否则，仅生成此Cell的缓冲区。默认 ``True`` 。

        返回：
            Iterator[Tensor]，缓冲区的迭代器。

    .. py:method:: cast_inputs(inputs, dst_type)

        将输入转换为指定类型。

        .. warning::
            此接口将在后续版本中废弃。

    .. py:method:: cells()

        返回当前Cell的子Cell的迭代器。

        返回：
            Iteration类型，Cell的子Cell。

    .. py:method:: cells_and_names(cells=None, name_prefix='')

        递归地获取当前Cell及输入 `cells` 的所有子Cell的迭代器，包括Cell的名称及其本身。

        参数：
            - **cells** (str) - 需要进行迭代的Cell。默认值： ``None`` 。
            - **name_prefix** (str) - 作用域。默认值： ``''`` 。

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

    .. py:method:: compiled
        :property:

        在图模式下使用，用于标记 `Cell` 是否已被编译。
    
    .. py:method:: construct(*args, **kwargs)

        定义要执行的计算逻辑。所有子类都必须重写此方法。

        .. note::
            当前不支持inputs同时输入tuple类型和非tuple类型。

        参数：
            - **args** (tuple) - 可变参数列表，默认值： ``()`` 。
            - **kwargs** (dict) - 可变的关键字参数的字典，默认值： ``{}`` 。

        返回：
            Tensor类型，返回计算结果。

    .. py:method:: extend_repr()

        在原有描述基础上扩展Cell的描述。

        若需要在print时输出个性化的扩展信息，请在您的网络中重新实现此方法。

    .. py:method:: generate_scope()

        为网络中的每个Cell对象生成NameSpace。

    .. py:method:: get_buffer(target)

        返回给定 `target` 的缓冲区，如果不存在则抛出错误。

        请参阅 `get_sub_cell` 的文档，了解有关此方法功能的更详细说明以及如何正确指定 `target`。

        参数：
            - **target** (str) - 要查找的缓冲区的完全限定字符串名称。（请参阅 `get_sub_cell` 了解如何指定完全限定字符串。）

        返回：
            Tensor

    .. py:method:: get_extra_state()

        返回要包含在Cell的 `state_dict` 中的任何额外状态。

        当构建Cell的 `state_dict()` 时，将调用此函数。
        如果您需要存储额外状态，实现此方法，并为您的Cell实现相应的 :func:`set_extra_state` 。

        请注意，额外状态应为可序列化对象（picklable），以确保state_dict的序列化可用性。
        仅对tensor的序列化提供向后兼容性保证；
        对于其他对象，如果其序列化的pickled形式发生变化，可能会导致向后兼容性问题。

        返回：
            object，要存储在Cell的state_dict中的额外状态。

    .. py:method:: get_flags()

        获取该Cell的自定义属性，自定义属性通过 `add_flags` 方法添加。

    .. py:method:: get_func_graph_proto()

        返回图的二进制原型。

    .. py:method:: get_inputs()

        返回编译计算图所设置的输入。

        返回：
            Tuple类型，编译计算图所设置的输入。

        .. warning::
            这是一个实验性API，后续可能修改或删除。

    .. py:method:: get_parameters(expand=True)

        返回Cell中parameter的迭代器。

        获取Cell的参数。如果 `expand` 为 ``true`` ，获取此cell和所有subcells的参数。关于subcell，请看下面的示例。

        参数：
            - **expand** (bool) - 如果为 ``True`` ，则递归地获取当前Cell和所有子Cell的parameter。否则，只生成当前Cell的subcell的parameter。默认值： ``True`` 。

        返回：
            Iteration类型，Cell的parameter。

    .. py:method:: get_scope()

        返回Cell的作用域。

        返回：
            String类型，网络的作用域。

    .. py:method:: get_sub_cell(target)

        返回给定 `target` 的子Cell，如果不存在则抛出错误。

        例如，假设你有一个 `nn.Cell` `A`，如下所示：

        .. code-block:: text

            A(
                (net_b): NetB(
                    (net_c): NetC(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (dense): Dense(in_features=100, out_features=200, bias=True)
                )
            )

        （该图显示了 `nn.Cell` `A` 。 `A` 有一个嵌套的子Cell `net_b`，
        而后者本身又有两个子Cell `net_c` 和 `dense` 。 `net_c` 则有一个子Cell `conv` 。）

        要检查是否拥有子Cell `dense` ，我们将调用 `get_sub_cell("net_b.dense")` 。要检查是否拥有子Cell `conv` ，我们将调用 `get_sub_cell("net_b.net_c.conv")` 。

        `get_sub_cell` 的运行时间受 `target` 中Cell嵌套程度的限制。使用 `name_cells` 的查询可获得相同的结果，但传递的Cell的数量级为O(N)。
        因此，为了简单检查是否存在某些子Cell，应始终使用 `get_sub_cell` 。

        参数：
            - **target** (str) - 要查找的子Cell的完全限定字符串名称。（请参阅上述示例以了解如何指定完全限定字符串。）

        返回：
            Cell

    .. py:method:: init_parameters_data(auto_parallel_mode=False)

        初始化并替换Cell中所有的parameter的值。

        .. note::
            在调用 `init_parameters_data` 后，`trainable_params()` 或其他相似的接口可能返回不同的参数对象，不建议保存这些结果。

        参数：
            - **auto_parallel_mode** (bool) - 是否在自动并行模式下执行。默认值： ``False`` 。

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
            - **check_name_contain_dot** (bool) - 是否对 `param_name` 中的"."进行检查。默认值： ``True`` 。

        异常：
            - **KeyError** - 如果参数名称为空或包含"."。
            - **TypeError** - 如果参数的类型不是Parameter。

    .. py:method:: load_state_dict(state_dict, strict=True)

        将 :attr:`state_dict` 中的参数和缓冲区复制到当前Cell及其子Cell中。

        如果 `strict` 设置为 ``True`` ，则 :attr:`state_dict` 的键必须与该Cell的 :func:`mindspore.nn.Cell.state_dict` 方法返回的键完全匹配。

        参数：
            - **state_dict** (dict) - 包含参数和持久缓冲区的字典。
            - **strict** (bool，可选) - 是否严格要求输入 `state_dict` 中的键必须与当前Cell的 :func:`mindspore.nn.Cell.state_dict` 方法返回的键匹配。默认 ``True`` 。

        返回：
            一个包含 `missing_keys` 和 `unexpected_keys` 字段的namedtuple，

            - `missing_keys` 是一个包含字符串的列表，表示当前Cell需要但在state_dict中缺失的键。

            - `unexpected_keys` 是一个包含字符串的列表，表示state_dict中存在但当前Cell不需要的键。

        .. note::
            如果 `strict` 为 ``True`` 且某个参数或缓冲区被注册为 ``None`` ，但其对应的键在 :attr:`state_dict` 中存在，则 :func:`mindspore.nn.Cell.load_state_dict` 将会抛出 ``RuntimeError`` 。

    .. py:method:: name_cells()

        递归地获取一个Cell中所有子Cell的迭代器。

        包括Cell名称和Cell本身。

        返回：
            Dict[String, Cell]，Cell中的所有子Cell及其名称。

    .. py:method:: named_buffers(prefix="", recurse=True, remove_duplicate=True)

        返回Cell中缓冲区的迭代器，包含缓冲区的名称以及缓冲区本身。

        参数：
            - **prefix** (str，可选) - 添加到所有缓冲区名称前面的前缀。默认 ``""`` 。
            - **recurse** (bool，可选) - 如果为 ``True`` ，则生成此Cell和所有子Cell的缓冲区。否则，仅生成此Cell的缓冲区。默认 ``True`` 。
            - **remove_duplicate** (bool，可选) - 是否删除结果中的重复缓冲区。默认 ``True`` 。

        返回：
            Iterator[Tuple[str, Tensor]]，包含名称和缓冲区的元组的迭代器。

    .. py:method:: offload(backward_prefetch="Auto")

        设置Cell激活值卸载，设置后该Cell中所有的Primitive类会被使能激活值卸载标签。若激活值需要在反向阶段被用于计算
        梯度，则该激活值会在正向阶段被搬运至host侧，不会存储在device侧，并在反向阶段使用其之前，预取回device侧。

        .. note::
            - 当某个Cell被标记为offload时，运行模型必须为"GRAPH_MODE"模式。
            - 当某个Cell被标记为offload时，需要开启lazyinline。

        参数：
            - **backward_prefetch** (Union[str, int]，可选) - 设置反向阶段提前预取激活值的时机。默认值： ``"Auto"`` 。当为 ``"Auto"`` 时，框架将提前一个算子开始预取激活值；当为正整数时，框架将提前 ``backward_prefetch`` 个算子开始预期激活值，例如1、20、100。

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
            - **name_prefix** (str) - 作用域。默认值： ``''`` 。
            - **expand** (bool) - 如果为True，则递归地获取当前Cell和所有子Cell的参数及名称；如果为 ``False`` ，只生成当前Cell的子Cell的参数及名称。默认值： ``True`` 。

        返回：
            迭代器，Cell的名称和Cell本身。

        教程样例：
            - `网络构建 - 模型参数 <https://mindspore.cn/tutorials/zh-CN/master/beginner/model.html#模型参数>`_

    .. py:method:: parameters_broadcast_dict(recurse=True)

        获取这个Cell的参数广播字典。

        参数：
            - **recurse** (bool) - 是否包含子Cell的参数。默认值： ``True`` 。

        返回：
            OrderedDict，返回参数广播字典。

    .. py:method:: parameters_dict(recurse=True)

        获取此Cell的parameter字典。

        参数：
            - **recurse** (bool) - 是否递归地包含所有子Cell的parameter。默认值： ``True`` 。

        返回：
            OrderedDict类型，返回参数字典。

    .. py:method:: pipeline_segment
        :property:

        `pipeline_segment` 表示当前Cell所在的segment。

    .. py:method:: pipeline_stage
        :property:

        `pipeline_stage` 表示当前Cell所在的stage。

    .. py:method:: recompute(*, use_reentrant=True, output_recompute=False, **kwargs)

        设置Cell重计算。Cell中输出算子以外的所有算子将被设置为重计算。如果一个算子的计算结果被输出到一些反向节点来进行梯度计算，且被设置成重计算，那么我们会在反向传播中重新计算它，而不去存储在前向传播中的中间激活层的计算结果。

        .. note::
            - 如果计算涉及到诸如随机化或全局变量之类的操作，那么目前还不能保证等价。
            - 如果该Cell中算子的重计算API也被调用，则该算子的重计算模式以算子的重计算API的设置为准。
            - 该接口仅配置一次，即当父Cell配置了，子Cell不需再配置。
            - Cell的输出算子默认不做重计算，这一点是基于我们减少内存占用的配置经验。如果一个Cell里面只有一个算子，且想要把这个算子设置为重计算的，那么请使用算子的重计算API。
            - 当应用了重计算且内存充足时，可以配置'mp_comm_recompute=False'来提升性能。
            - 当应用了重计算但内存不足时，可以配置'parallel_optimizer_comm_recompute=True'来节省内存。有相同融合group的Cell应该配置相同的parallel_optimizer_comm_recompute。

        关键字参数：
            - **use_reentrant** (bool，可选) - 该参数只在PyNative模式下有效。若设置为 ``True``，将通过自定义反向传播函数实现重计算，该方式不支持List/Tuple等复杂类型的求导；若设置为 ``False``，将使用 :class:`mindspore.saved_tensors_hooks` 实现重计算，该方式支持对复杂类型内部张量的求导。默认值： ``True`` 。
            - **output_recompute** (bool，可选) - 该参数只在PyNative模式下有效。若设置为 ``True``，默认使用 :class:`mindspore.saved_tensors_hooks` 功能实现重计算。该模块的输出不会被后续需要求导的算子缓存。当存在两个相邻cell均需重计算时（其中一个cell的输出作为另一个cell的输入），这两个cell的重计算将被融合。在此情况下，第一个cell的输出激活值将不会被保存。默认值： ``False`` 。
            - **\*\*kwargs** - 其他参数。

              - **mp_comm_recompute** (bool，可选) - 表示在自动并行或半自动并行模式下，指定Cell内部由模型并行引入的通信操作是否重计算。默认值： ``True`` 。
              - **parallel_optimizer_comm_recompute** (bool，可选) - 表示在自动并行或半自动并行模式下，指定Cell内部由优化器并行引入的AllGather通信是否重计算。默认值： ``False`` 。

    .. py:method:: register_backward_hook(hook_fn)

        设置Cell对象的反向hook函数。

        .. note::
            - hook_fn必须有如下代码定义：`cell` 是已注册Cell对象的信息， `grad_input` 是Cell对象的反向输出梯度， `grad_output` 是反向传递给Cell对象的梯度。 用户可以在hook_fn中返回None或者返回新的梯度。
            - hook_fn返回None或者新的相应于 `grad_input` 的梯度：hook_fn(cell, grad_input, grad_output) -> New grad_input or None。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用 `register_backward_hook(hook_fn)` 。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用 `register_backward_hook(hook_fn)` ，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和反向输入，输出梯度的 `hook_fn` 函数。

        返回：
            返回与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_backward_pre_hook(hook_fn)

        设置Cell对象的反向pre_hook函数。

        .. note::
            - hook_fn必须有如下代码定义：`cell` 是已注册Cell对象的信息， `grad_output` 是反向传递给Cell对象的梯度。用户可以在hook_fn中返回None或者返回新的梯度。
            - hook_fn返回None或者新的相应于 `grad_output` 的梯度：hook_fn(cell, grad_output) -> New grad_output or None。
            - `register_backward_pre_hook(hook_fn)` 在PyThon环境中运行。为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用 `register_backward_pre_hook(hook_fn)` 。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用 `register_backward_pre_hook(hook_fn)` ，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和反向输入梯度的 `hook_fn` 函数。

        返回：
            返回与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_buffer(name, tensor, persistent=True)

        在Cell添加一个缓冲区 `buffer` 。

        这通常用于注册不应被视为模型参数的缓冲区。例如，BatchNorm的 `running_mean` 不是参数，而是Cell状态的一部分。
        默认情况下，缓冲区是持久的，将与参数一起保存。可以通过将 `persistent` 设置为 ``False`` 来更改此行为。
        持久缓冲区和非持久缓冲区之间的唯一区别是后者不会成为此Cell的 :attr:`state_dict` 的一部分。

        可以使用指定的名称将缓冲区作为属性访问。

        参数：
            - **name** (str) - 缓冲区的名字。可以使用给定的名称访问此Cell的缓冲区 。
            - **tensor** (Tensor) - 待注册的缓冲区。如果为 ``None`` ，则此Cell的 :attr:`state_dict` 不会包括该缓冲区。
            - **persistent** (bool, 可选) - 缓冲区是否是此Cell的 :attr:`state_dict` 的一部分。默认 ``True`` 。

    .. py:method:: register_forward_hook(hook_fn, with_kwargs=False)

        设置Cell对象的正向hook函数。

        该hook函数会在 :func:`mindspore.nn.Cell.construct` 执行并生成输出之后被调用。

        `hook_fn` 必须符合以下两种函数签名之一：

        - 当 `with_kwargs` 为 ``False`` 时，`hook_fn(cell, args, output) -> None or new_output` 。
        - 当 `with_kwargs` 为 ``True`` 时，`hook_fn(cell, args, kwargs, output) -> None or new_output` 。

        其中：

        - `cell` (Cell)：注册hook的Cell对象。
        - `args` (tuple)：传递给 `construct` 函数的位置参数。
        - `kwargs` (dict)：传递给 `construct` 函数的关键字参数。仅当 `with_kwargs` 为 ``True`` 时，这些参数才会传递给 `hook_fn` 。
        - `output` ： `construct` 函数生成的输出。

        .. note::
            - `hook_fn` 可以通过返回新的输出数据来修改前向输出。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用此方法。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用此方法，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和正向输入，输出数据的 `hook_fn` 函数。
            - **with_kwargs** (bool，可选) - 是否将 `construct` 的关键字参数传递给hook函数。默认值： ``False`` 。

        返回：
            返回与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_forward_pre_hook(hook_fn, with_kwargs=False)

        设置Cell对象的正向pre_hook函数。

        该hook函数会在 :func:`mindspore.nn.Cell.construct` 执行前调用。

        hook 函数需满足以下两种签名之一：

        - 当 `with_kwargs` 为 ``False`` 时， `hook_fn(cell, args) -> None or new_args` 。
        - 当 `with_kwargs` 为 ``True`` 时， `hook_fn(cell, args, kwargs) -> None or (new_args, new_kwargs)` 。

        其中：

        - `cell` (Cell)：注册hook的Cell对象。
        - `args` (tuple)：传入 `construct` 函数的位置参数。
        - `kwargs` (dict)：传入 `construct` 函数的关键字参数。仅当 `with_kwargs` 为 ``True`` 时，这些参数才会传递给 `hook_fn` 。

        .. note::
            - `hook_fn` 可通过返回新的输入数据来修改前向输入。
              如果 `with_kwargs` 为 ``False`` ，可以返回单独的值（如果返回值不是元组，将自动封装为元组），也可以直接返回一个元组形式的参数列表。
              如果 `with_kwargs` 为 ``True`` ，则应该返回包含新的 `args` 和 `kwargs` 的元组。
            - 为了避免脚本在切换到图模式时运行失败，不建议在Cell对象的 `construct` 函数中调用此方法。
            - PyNative模式下，如果在Cell对象的 `construct` 函数中调用此方法，那么Cell对象每次运行都将增加一个 `hook_fn` 。

        参数：
            - **hook_fn** (function) - 捕获Cell对象信息和正向输入数据的hook_fn函数。
            - **with_kwargs** (bool，可选) - 是否将 `construct` 的关键字参数传递给hook函数。默认值： ``False`` 。

        返回：
            返回与 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

        异常：
            - **TypeError** - 如果 `hook_fn` 不是Python函数。

    .. py:method:: register_load_state_dict_post_hook(hook)

        为 :func:`mindspore.nn.Cell.load_state_dict` 方法注册一个后钩子。

        它应该具有以下签名:

        hook(cell, incompatible_keys) -> None

        参数 `cell` 是此钩子注册的当前cell，参数 `incompatible_keys` 是一个 `NamedTuple` ，由属性 `missing_keys` 和 `unexpected_keys` 组成。`missing_keys` 是包含缺失键的 `list` ，
        而 `unexpected_keys` 是包含意外键的 `list` 。

        请注意，正如预期的那样，在使用 `strict=True` 调用：func: `load_state_dict` 时执行的检查会受到钩子对 `missing_keys` 或 `unexpected_keys` 所做修改的影响。
        当 `strict=True` 时，添加任何一组键都会导致抛出错误，而清除缺失和意外的键将避免错误。

        参数：
            - **hook** (Callable) - 在调用load_state_dict之后执行的钩子。

        返回：
            一个句柄，可以通过调用 `handle.remove()` 来移除已添加的钩子。

    .. py:method:: register_load_state_dict_pre_hook(hook)

        为 :func:`mindspore.nn.Cell.load_state_dict` 方法注册一个预钩子。

        它应该具有以下签名:

        hook(cell, state_dict, prefix, local_metadata, strict, missing_keys, expected_keys, error_msgs) -> None

        注册的钩子可以就地修改 `state_dict` 。

        参数：
            - **hook** (Callable) - 在调用load_state_dict之前执行的钩子。

        返回：
            一个句柄，可以通过调用 `handle.remove()` 来移除已添加的钩子。

    .. py:method:: register_parameter_forward_hook(hook)

        为当前Cell实例注册Parameter Forward Hook。

        此函数可以将Parameter Forward Hook注册到整个网络（当 `hook` 为可调用对象时），或者注册到特定的Parameter上（当 `hook` 为指定Parameter-Hook对的字典列表时）。

        .. warning::

            这是一个实验性API，后续可能修改或删除。

        .. note::
            - 此API仅支持图模式或被 `jit` 装饰的函数。
            - 当Parameter在不同 Cell 之间共享时，不支持Parameter Forward Hook。
              以下是一个 **不支持** 的模式示例：

              .. code-block:: python

                  class InnerBlock(nn.Cell):
                      def __init__(self):
                          super().__init__()
                          self.shared_weight = ms.Parameter(ms.Tensor([1.0]), name="shared_weight")
                      def construct(self, x):
                          return x * self.shared_weight

                  class OuterNet(nn.Cell):
                      def __init__(self):
                          super().__init__()
                          self.inner = InnerBlock()
                      def construct(self, x):
                          out = self.inner(x)
                          # 跨Cell的Parameter共享（Parameter Forward Hook不支持此场景）
                          out = out * self.inner.shared_weight
                          return out

        参数：
            - **hook** (Union[Callable, List[Dict[str, Any]]]) - 要注册的Hook或Hook列表。

              - 如果是一个可调用函数，它将被设置为整个网络范围（全局）的Parameter Forward Hook。
              - 如果是一个字典列表，则每个字典必须包含且仅包含两个键：

                - ``"params"`` - 一个 :class:`mindspore.Parameter` 对象组成的列表。
                - ``"hook"`` - 一个可调用的Hook函数。该Hook将被独立地注册到每个指定的Parameter上。

        异常：
            - **ValueError** - 如果Hook列表中的任何字典没有同时包含 ``"params"`` 和 ``"hook"`` 键。
            - **TypeError** - 如果 ``"params"`` 列表中的任何项不是 :class:`mindspore.Parameter` 类型，或者 ``"hook"`` 的值不是可调用的。

    .. py:method:: register_saved_tensors_hooks(pack_hook, unpack_hook)

        注册用于处理保存张量（Saved Tensor）的打包（pack）和解包（unpack）钩子函数。

        其作用范围限定在 :func:`mindspore.nn.Cell.construct` 内，更多使用说明请参考 :class:`mindspore.saved_tensors_hooks` 。

        参数：
            - **pack_hook** (Callable) - 定义前向计算保存张量时的处理方法。
            - **unpack_hook** (Callable) - 定义反向计算恢复张量时的处理方法。

    .. py:method:: register_state_dict_post_hook(hook)

        为 :func:`mindspore.nn.Cell.state_dict` 方法注册一个后钩子。

        它应该具有以下签名:

        hook(cell, state_dict, prefix, local_metadata) -> None

        注册的钩子可用于在调用 `state_dict` 之后执行后处理。

        参数：
            - **hook** (Callable) - 在调用state_dict之后执行的钩子。

        返回：
            一个句柄，可以通过调用 `handle.remove()` 来移除已添加的钩子。

    .. py:method:: register_state_dict_pre_hook(hook)

        为 :func:`mindspore.nn.Cell.state_dict` 方法注册一个预钩子。

        它应该具有以下签名:

        hook(cell, prefix, keep_vars) -> None

        注册的钩子可用于在调用 `state_dict` 之前执行预处理。

        参数：
            - **hook** (Callable) - 在调用state_dict之前执行的钩子。

        返回：
            一个句柄，可以通过调用 `handle.remove()` 来移除已添加的钩子。

    .. py:method:: remove_redundant_parameters()

        删除冗余参数。

        .. warning::
            此接口将在后续版本中废弃。

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
        `algorithm library <https://atomgit.com/mindspore/mindspore/tree/master/mindspore/python/mindspore/boost>`_ 算法库中。

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
            - **mode** (bool) - 指定当前模式是否进行参数广播。默认值： ``True`` 。

    .. py:method:: set_comm_fusion(fusion_type, recurse=True)

        为Cell中的参数设置融合类型。请参考 :class:`mindspore.Parameter.comm_fusion` 的描述。

        .. note:: 当函数被多次调用时，此属性值将被重写。

        参数：
            - **fusion_type** (int) - Parameter的 `comm_fusion` 属性的设置值。
            - **recurse** (bool) - 是否递归地设置子Cell的可训练参数。默认值： ``True`` 。

    .. py:method:: set_data_parallel()

        在非自动策略搜索的情况下，如果此Cell的所有算子（包括此Cell内含嵌套的cell）未指定并行策略，则将为这些基本算子设置为数据并行策略。

        .. note:: 仅在图模式，使用auto_parallel_context = ParallelMode.AUTO_PARALLEL生效。

    .. py:method:: set_extra_state(state)

        设置加载的 `state_dict` 中包含的额外状态。

        此方法由 `load_state_dict` 调用，以处理 `state_dict` 中的任何额外状态。
        如果您的 Cell 需要在 `state_dict` 中存储额外状态，请实现此方法及相应的
        `get_extra_state` 方法。

        参数：
            - **state** (dict) - `state_dict` 的额外状态。

    .. py:method:: set_grad(requires_grad=True)

        Cell的梯度设置。

        参数：
            - **requires_grad** (bool) - 指定网络是否需要梯度，如果为 ``True`` ，PyNative模式下Cell将构建反向网络。默认值： ``True`` 。

        返回：
            Cell类型，Cell本身。

    .. py:method:: set_inputs(*inputs, **kwargs)

        设置编译计算图所需的输入。输入数量需与数据集数量一致。若使用Model接口，请确保所有传入Model的网络和损失函数都配置了set_inputs。
        输入Tensor的shape可以为动态或静态。

        .. note::
            有两种配置模式：

            - 全量配置模式：输入将被用作图编译时的完整编译参数。
            - 增量配置模式：输入被配置到Cell的部分输入上，这些输入将替换图编译对应位置上的参数。

            只能传入inputs和kwargs的其中一个。inputs用于全量配置模式，kwargs用于增量配置模式。

        参数：
            - **inputs** (tuple) - 全量配置模式的参数。
            - **kwargs** (dict) - 增量配置模式的参数。可设置的key值为 `self.construct` 中定义的参数名。

        .. warning::
            这是一个实验性API，后续可能修改或删除。

    .. py:method:: set_jit_config(jit_config)

        为Cell设置编译时所使用的JitConfig配置项。

        参数：
            - **jit_config** (JitConfig) - Cell的Jit配置信息。详情请参考 :class:`mindspore.JitConfig` 。

    .. py:method:: set_train(mode=True)

        将Cell设置为训练模式。

        设置当前Cell和所有子Cell的训练模式。对于训练和预测具有不同结构的网络层(如 `BatchNorm`)，将通过这个属性区分分支。如果设置为True，则执行训练分支，否则执行另一个分支。

        .. note::
            当执行 :func:`mindspore.train.Model.train` 的时候，框架会默认调用Cell.set_train(True)。
            当执行 :func:`mindspore.train.Model.eval` 的时候，框架会默认调用Cell.set_train(False)。

        参数：
            - **mode** (bool) - 指定模型是否为训练模式。默认值： ``True`` 。

        返回：
            Cell类型，Cell本身。

        教程样例：
            - `模型训练 - 训练与评估实现 <https://mindspore.cn/tutorials/zh-CN/master/beginner/train.html#训练与评估>`_

    .. py:method:: shard(in_strategy, out_strategy=None, parameter_plan=None)

        指定输入/输出Tensor的分布策略，通过其余算子的策略推导得到。在图模式下，
        可以利用此方法设置某个模块的分布式切分策略，未设置的会自动通过策略传播方式配置。 in_strategy/out_strategy需要为元组类型，
        其中的每一个元素指定对应的输入/输出的Tensor分布策略，可参考： :func:`mindspore.ops.Primitive.shard` 的描述。
        其余算子的并行策略由输入输出指定的策略推导得到。

        .. note:: 
            - 仅在半自动并行或自动并行模式下有效。在其他并行模式中，将忽略此处设置的策略。
            - 如果输入含有Parameter，其对应的策略应该在 `in_strategy` 里设置。

        .. warning::
            该方法当前不支持在PyNative模式下使用。

        参数：
            - **in_strategy** (tuple) - 指定各输入的切分策略，输入元组的每个元素元组，元组即具体指定输入每一维的切分策略。
            - **out_strategy** (Union[None, tuple]) - 指定各输出的切分策略，用法同in_strategy。默认值： ``None`` 。
            - **parameter_plan** (Union[dict, None]) - 指定各参数的切分策略，传入字典时，键是str类型的参数名，值是一维整数tuple表示相应的切分策略，
              如果参数名错误或对应参数已经设置了切分策略，该参数的设置会被跳过。默认值： ``None`` 。

    .. py:method:: state_dict(*args, destination=None, prefix="", keep_vars=False)

        返回一个包含对Cell整个状态的引用的字典。

        参数和持久缓冲区（例如运行平均值）都包括在内。键是相应的参数和缓冲区名称。设置为 `None` 的参数和缓冲区不包括在内。

        .. note::
            返回的对象是一个浅拷贝。它包含对该Cell的参数和缓冲区的引用。

        .. warning::
            - 目前 `state_dict()` 还按顺序接受 `destination` 、`prefix` 和 `keep_vars` 的位置参数。但是这即将被弃用，关键字参数将在未来的版本中强制执行。
            - 请避免使用参数 `destination` ，因为它不是为最终用户设计的。

        参数：
            - **destination** (dict，可选) - 如果提供，Cell的状态将更新到此字典中，并返回相同的对象。否则，将创建并返回 `OrderedDict` 。默认 ``None`` 。
            - **prefix** (str，可选) - 添加到参数和缓冲区名称的前缀，用于组成state_dict中的键。默认 ``""`` 。
            - **keep_vars** (bool，可选) - 状态字典返回值是否为拷贝。默认 ``False`` ，返回引用。

        返回：
            Dict，包含整个Cell状态的字典。

    .. py:method:: to_float(dst_type)

        在Cell和所有子Cell的输入上添加类型转换，以使用特定的浮点类型运行。

        如果 `dst_type` 是 `mindspore.dtype.float16` ，Cell的所有输入(包括作为常量的input、Parameter、Tensor)都会被转换为float16。请参考 :func:`mindspore.amp.build_train_network` 的源代码中的用法。

        .. note:: 多次调用将产生覆盖。

        参数：
            - **dst_type** (mindspore.dtype) - Cell转换为 `dst_type` 类型运行。 `dst_type` 可以是 `mindspore.dtype.float16` 、 `mindspore.dtype.float32` 或者  `mindspore.dtype.bfloat16` 。

        返回：
            Cell类型，Cell本身。

        异常：
            - **ValueError** - 如果 `dst_type` 不是 `mindspore.dtype.float32` ，不是 `mindspore.dtype.float16` , 也不是 `mindspore.dtype.bfloat16` 。

    .. py:method:: trainable_params(recurse=True)

        返回Cell的一个可训练参数的列表。

        参数：
            - **recurse** (bool) - 是否递归地包含当前Cell的所有子Cell的可训练参数。默认值： ``True`` 。

        返回：
            List类型，可训练参数列表。

        教程样例：
            - `模型训练 - 优化器 <https://mindspore.cn/tutorials/zh-CN/master/beginner/train.html#优化器>`_

    .. py:method:: untrainable_params(recurse=True)

        返回Cell的一个不可训练参数的列表。

        参数：
            - **recurse** (bool) - 是否递归地包含当前Cell的所有子Cell的不可训练参数。默认值： ``True`` 。

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
            - **prefix** (str) - 前缀字符串。默认值： ``''`` 。
            - **recurse** (bool) - 是否递归地包含所有子Cell的参数。默认值： ``True`` 。
