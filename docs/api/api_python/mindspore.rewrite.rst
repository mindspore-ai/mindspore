mindspore.rewrite
=================
MindSpore的ReWrite功能用于修改网络前向的计算过程，对网络进行插入、删除和替换语句。该功能目前处于开发调试阶段，可能会更改或删除。

ReWrite完整示例请参考
`rewrite_example.py <https://gitee.com/mindspore/mindspore/tree/master/docs/api/api_python/rewrite_example.py>`_ 。
该样例代码的主要功能包括：怎么通过网络创建SymbolTree，并且对SymbolTree中的节点进行插入删除替换等操作，其中还包含了对子网络的修改和通过模式匹配进行节点替换。

.. literalinclude:: rewrite_example.py
    :language: python
    :start-at: import

.. py:class:: mindspore.rewrite.SymbolTree(handler: SymbolTreeImpl)

    SymbolTree通常对应于网络的前向计算过程。

    参数：
        - **handler** (SymbolTreeImpl) - SymbolTree内部实现实例。

    .. py:method:: mindspore.rewrite.SymbolTree.after(node: Node)

        获取插入位置，位置为 `node` 之后。
        返回值用于指示插入节点的位置，它指示在源代码中的位置，而不是在拓扑顺序中的位置。不需要关心 Position是什么，只需将其视为处理程序并将其用作SymbolTree的插入接口的参数。

        参数：
            - **node** (Node) - 指定插入位置在哪个节点之后，可以是Node或者Node的名称。

        返回：
            Position，指定插入节点的位置。

        异常：
            - **TypeError** - 参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.before(node: Node)

        与after的区别是，该接口返回的位置为 `node` 之前。
        返回值用于指示插入节点的位置，它指示在源代码中的位置，而不是在拓扑顺序中的位置。不需要关心 `Position` 是什么，只需将其视为处理程序并将其用作 `SymbolTree` 的插入接口的参数。

        参数：
            - **node** (Node) - 指定插入位置在哪个节点之前，可以是Node或者Node的名称。

        返回：
            Position，指定插入节点的位置。

        异常：
            - **TypeError** - 参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.create(network)

        根据传入的 `network` 创建SymbolTree对象。

        参数：
            - **network** (Cell) - 重写的网络。

        返回：
            SymbolTree，基于 `network` 创建的符号树。

        异常：
            - **TypeError** - 参数 `network` 不是Cell类型对象。

    .. py:method:: mindspore.rewrite.SymbolTree.dump()

        将 `SymbolTree` 中network对应的ir图信息打印到屏幕。

    .. py:method:: mindspore.rewrite.SymbolTree.erase_node(node: Node)

        删除SymbolTree中的一个节点。被删除的节点必须不被其他节点依赖。

        参数：
            - **node** (Node) - 被删除的节点。可以是Node或者Node的名称。

        返回：
            如果 `node` 属于当前的SymbolTree则返回被删除节点。否则返回None。

        异常：
            - **TypeError** - 如果参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.get_code()

        获取SymbolTree所对应的源代码。

        返回：
            str，SymbolTree对应的源码字符串。

    .. py:method:: mindspore.rewrite.SymbolTree.get_network()

        获取SymbolTree所对应的生成的网络对象。源码会保存到文件中，默认的文件名为 `network_define.py`。

        返回：
            根据SymbolTree生成的网络对象。

    .. py:method:: mindspore.rewrite.SymbolTree.insert(position, node: Node)

        在SymbolTree的 `position` 位置插入一个节点。 `position` 可以通过 `before` 或 `after` 来获得。

        参数：
            - **position** (Position) - 插入位置。
            - **node** (Node) - 要插入的节点。

        返回：
            `Node`，被插入的节点, 当调用此方法时会对参数进行唯一性处理， `node` 会被修改。

        异常：
            - **RuntimeError** - 如果 `position` 指定的不是该SymbolTree内的位置。
            - **TypeError** - 如果参数 `position` 不是Position类型。
            - **TypeError** - 如果参数 `node` 不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.nodes()

        获取当前SymbolTree的节点，用于遍历。

        返回：
            当前SymbolTree中节点的生成器。

    .. py:method:: mindspore.rewrite.SymbolTree.replace(old_node: Node, new_nodes: [Node])

        使用新节点列表来替代旧节点。

        .. note::
            - 仅支持一对一更换或一对多替换。如果需要多对多替换，请参考PatternEngine。
            - 当一对多替换时，Rewrite会将 `new_nodes` 中所有节点插入到 `symbol_tree` 中。
            - 调用者应指定子树内节点的参数和输出来确定子树内的拓扑关系。
            - 调用者应指定子树输入节点的参数来确定子树与原始树中节点的拓扑关系。
            - ReWrite将维护子树的前置节点的参数，用于指定子树输出的拓扑关系。
            - 将 `new_nodes` 替换到SymbolTree后，ReWrite将维护节点的所有输入。

        参数：
            - **old_node** (Node) - 被替换节点。
            - **new_nodes** (list[Node]) - 要替换进SymbolTree的节点列表。

        返回：
            替换到SymbolTree的节点列表的根节点。

        异常：
            - **RuntimeError** - 如果 `old_node` 仍然被其他节点依赖。
            - **TypeError** - 如果参数 `new_nodes` 不是list，或者列表中的成员不是Node类型。
            - **TypeError** - 如果参数 `old_node` 不是Node类型。

.. py:class:: mindspore.rewrite.Node(node: NodeImpl)

    节点是表达网络中源代码的一种数据结构。

    在大多数情况下，Node表示前向计算的的运算，它可以是Cell的实例、Primitive的实例或可调用的方法。

    参数：
        - **node** (NodeImpl) - `NodeImpl` 的handle。NodeImpl是Node的实现，不是Rewrite的接口。Rewrite建议调用Node的特定 `create` 方法来实例化Node的实例，例如 `create_call_cell`，而不直接调用Node的构造函数，不需关心NodeImpl是什么，只需作为handle看待。

    .. py:method:: mindspore.rewrite.Node.create_call_cell(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None, name: str = "", is_sub_net: bool = False)
        :staticmethod:

        通过该接口可以根据 `cell` 对象创建一个Node实例。节点对应的源代码格式：
        ``targets = self.name(*args, **kwargs)``。

        参数：
            - **cell** (Cell) - 该节点对应的前向计算的Cell对象。
            - **targets** (list[ScopedValue]) - 表示输出名称。在源代码中作为节点的输出。Rewrite将在插入节点时检查并确保每个目标的唯一性。
            - **args** (list[ScopedValue]) - 该节点的参数名称。用作源代码中代码语句的参数。表示 `cell` 没有参数输入。Rewrite将在插入节点时检查并确保每个 `arg` 的唯一性。默认值：None。
            - **kwargs** (dict) - 键的类型必须是str，值的类型必须是ScopedValue。用来说明带有关键字的形参的输入参数名称。输入名称在源代码中作为语句表达式中的 `kwargs`。表示 `cell` 没有 `kwargs` 输入。Rewrite将在插入节点时检查并确保每个 `kwarg` 的唯一性。默认值：None。
            - **name** (str) - 表示节点的名称。用作源代码中的字段名称。当名称为无时，ReWrite将根据 `target` 生成一个默认名称。Rewrite将在插入节点时检查并确保名称的唯一性。默认值：None。
            - **is_sub_net** (bool) - 表示 `cell` 是否是一个网络。如果 `is_sub_net` 为真，Rewrite将尝试将 `cell` 解析为TreeNode，否则为CallCell节点。默认值：False。

        返回：
            Node实例。

        异常：
            - **TypeError** - 如果参数 `cell` 不是Cell类型。
            - **TypeError** - 如果参数 `targets` 不是list类型。
            - **TypeError** - 如果参数 `targets` 的成员不是str或者ScopedValue类型。
            - **TypeError** - 如果参数 `args` 不是ScopedValue类型。
            - **TypeError** - 如果参数 `kwarg` 的 `key` 不是str类型或者 `value` 不是ScopedValue类型。

    .. py:method:: mindspore.rewrite.Node.get_inputs()

        获取当前节点的拓扑序的输入节点。

        返回：
            Node的实例列表。

    .. py:method:: mindspore.rewrite.Node.get_instance_type()

        获取当前节点对应的 `operation` 实例类型。

        - 如果当前节点的 `node_type` 是 `CallCell`，该节点是Cell对象。
        - 如果当前节点的 `node_type` 是 `CallPrimitive`，该节点的是Primitive对象。
        - 如果当前节点的 `node_type` 是 `Tree`，该节点的类型是网络。
        - 如果当前节点的 `node_type` 是 `Python`、 `Input`、 `Output`、 `CallMethod`，该节点的类型为NoneType。

        返回：
            当前节点的 `operation` 类型。

    .. py:method:: mindspore.rewrite.Node.get_name()

        获取当前节点的名称。当节点被插入到SymbolTree时，节点的名称在SymbolTree中应该是唯一的。

        返回：
            节点的名称，类型为str。

    .. py:method:: mindspore.rewrite.Node.get_node_type()

        获取当前节点节点的类型。

        返回：
            NodeType，当前节点的类型。

    .. py:method:: mindspore.rewrite.Node.get_users()

        按拓扑顺序获取当前节点的输出节点。

        返回：
            输出节点的列表。

    .. py:method:: mindspore.rewrite.Node.set_arg(index: int, arg: Union[ScopedValue, str])

        设置当前节点的输入参数。

        参数：
            - **index** (int) - 要设置的参数索引。
            - **arg** (Union[ScopedValue, str]) - 新参数的值。

        异常：
            - **TypeError** - 如果参数 `index` 不是int类型。
            - **TypeError** - 如果参数 `arg` 不是str或者ScopedValue类型。

    .. py:method:: mindspore.rewrite.Node.set_arg_by_node(arg_idx: int, src_node: 'Node', out_idx: Optional[int] = None)

        将另一个节点设置为当前节点的输入。

        参数：
            - **arg_idx** (int) - 要设置的参数索引。
            - **src_node** (Node) - 输入的节点。
            - **out_idx** (int，optional) - 指定输入节点的哪个输出作为当前节点输入，则取第一个输出。默认值：None。

        异常：
            - **RuntimeError** - 如果 `src_node` 不属于当前的SymbolTree。
            - **TypeError** - 如果参数 `arg_idx` 不是int类型。
            - **ValueError** - 如果参数 `arg_idx` 超出了当前节点的参数数量。
            - **TypeError** - 如果参数 `src_node` 不是Node类型。
            - **TypeError** - 如果参数 `out_idx` 不是int类型。
            - **ValueError** - 如果参数 `out_idx` 超出了 `src_node` 的输出数量。
            - **ValueError** - 当 `out_idx` 为None或者没有给 `out_idx` 赋值时，参数 `src_node` 有多个输出。

.. py:class:: mindspore.rewrite.NodeType

    NodeType表示Node的类型。

    - **Unknown**：未初始化的节点类型。
    - **CallCell**： `CallCell` 节点表示在前向计算中调用Cell对象。
    - **CallPrimitive**： `CallPrimitive` 节点代表在前向计算中调用Primitive对象。
    - **CallMethod**： `CallMethod` 不能对应到Cell或者Primitive的节点。
    - **Python**： `Python` 节点包含不支持的 `ast` 的节点类型或不必要的解析 `ast` 节点。
    - **Input**：输入节点代表SymbolTree的输入，对应方法的参数。
    - **Output**: 输出节点代表SymbolTree的输出，对应方法的 `return` 语句。
    - **Tree**: 树节点代表转发方法中的子网调用。

.. py:class:: mindspore.rewrite.ScopedValue(arg_type: ValueType, scope: str = "", value=None)

    ScopedValue表示具有完整范围的值。

    ScopedValue用于表示：左值，如赋值语句的目标，或可调用对象，如调用语句的 `func`，或右值，如赋值语句的 `args` 和 `kwargs`。

    参数：
        - **arg_type** (ValueType) - 当前值的类型。
        - **scope** (str) - 字符串表示当前值的范围。以"self.var1"为例，这个var1的作用域是"self"。默认值： ""。
        - **value** - 当前ScopedValue中保存的值。值的类型对应于 `arg_type`。默认值：None。

    .. py:method:: mindspore.rewrite.ScopedValue.create_name_values(names: Union[list, tuple], scopes: Union[list, tuple] = None)
        :staticmethod:

        创建ScopedValue的列表。

        参数：
            - **names** (list[str] or tuple[str]) - 引用变量的名称，类型为str的列表或元组。
            - **scopes** (list[str] or tuple[str]) - 引用变量的范围，类型为str的列表或元组。表示没有指定作用范围。默认值：None。

        返回：
            ScopedValue的实例列表。

        异常：
            - **TypeError** - 如果 `names` 不是 `list` 或 `tuple` 或者其中的元素不是str类型。
            - **TypeError** - 如果 `scopes` 不是 `list` 或 `tuple` 或者其中的元素不是str类型。
            - **RuntimeError** - 如果 `names` 的长度不等于 `scopes` 的长度，而作用域不是None。

    .. py:method:: mindspore.rewrite.ScopedValue.create_naming_value(name: str, scope: str = "")

        创建一个使用变量名称命名的ScopedValue。NamingValue表示对另一个变量的引用。

        参数：
            - **name** (str) – 表示变量的字符串。
            - **scope** (str) – 表示变量范围的字符串，表示没有指定作用范围。默认值：空字符串。

        返回：
            ScopedValue的实例。

        异常：
            - **TypeError** - 如果 `name` 不是str类型。
            - **TypeError** - 如果 `scope` 不是str类型。

    .. py:method:: mindspore.rewrite.ScopedValue.create_variable_value(value)

        创建一个保存变量的ScopedValue。ScopedValue的类型由值的类型决定。ScopedValue的范围是空的。

        参数：
            - **value** - 要转换为ScopedValue的值。

        返回：
            ScopedValue的实例。

.. py:class:: mindspore.rewrite.ValueType

    ValueType表示ScopedValue的类型。

    - NamingValue表示对另一个变量的引用。
    - CustomObjValue表示自定义类的实例，或类型超出ValueType的基本类型和容器类型范围的对象。

.. py:class:: mindspore.rewrite.PatternEngine(pattern: Union[PatternNode, List], replacement: Replacement = None)

    PatternEngine通过PattenNode修改SymbolTree。

    参数：
        - **pattern** (Union[PatternNode, List]) - PatternNode的实例或用于构造 `Pattent` 的Cell类型列表。
        - **replacement** (callable) - 生成新节点的接口实现。

    .. py:method:: mindspore.rewrite.PatternEngine.apply(stree: SymbolTree)

        在 `stree` 上面执行当前的匹配模式。

        .. note::
            当前还不支持子树节点。

        参数：
            - **stree** (SymbolTree) - 要修改的SymbolTree。

        返回：
            bool，表示是否对 `stree` 进行了修改。

        异常：
            - **TypeError** - 如果参数 `stree` 不是SymbolTree类型。

    .. py:method:: mindspore.rewrite.PatternEngine.pattern()

        获取当前的匹配模式。

        返回：
            PattenNode的实例，用来说明当前模式需要匹配的类型。

.. py:class:: mindspore.rewrite.PatternNode(pattern_node_name: str, match_type: Type = Type[None], inputs: ['PatternNode'] = None)

    PatternNode在定义 `pattern` 时被定义为一个节点。

    参数：
        - **pattern_node_name** (str) - 节点名称。
        - **match_type** (Type) - 当前节点的匹配类型。默认值：None。
        - **inputs** (list[PatternNode]) - 当前节点的输入节点。默认值：None。

    .. py:method:: mindspore.rewrite.PatternNode.add_input(node)

        为当前节点添加输入。

        参数：
            - **node** (PatternNode) - 新增的输入节点。

        异常：
            - **TypeError** - 如果参数 `node` 不是PattenNode类型。

    .. py:method:: mindspore.rewrite.PatternNode.create_pattern_from_list(type_list: [])
        :staticmethod:

        使用类型的列表来创建Pattern。

        参数：
            - **type_list** (list[type]) - 类型列表。

        返回：
            根据列表生成的模式的根节点。

        异常：
            - **TypeError** - 如果 `type_list` 不是list类型。

    .. py:method:: mindspore.rewrite.PatternNode.create_pattern_from_node(node: Node)
        :staticmethod:

        根据节点及其输入创建Pattern。

        参数：
            - **node** (Node) - 要修改的节点。

        返回：
            根据 `node` 创建的PattentNode。

        异常：
            - **TypeError** - 如果 `node` 不是Node类型。

    .. py:method:: mindspore.rewrite.PatternNode.from_node(node: Node)
        :staticmethod:

        根据 `node` 创建PatternNode。

        参数：
            - **node** (Node) - 要修改的节点。

        返回：
            根据 `node` 创建的PattentNode。

        异常：
            - **TypeError** - 如果 `node` 不是Node类型。


    .. py:method:: mindspore.rewrite.PatternNode.get_inputs()

        获取当前节点的输入。

        返回：
            PattenNode的实例列表，当前节点的输入节点。

    .. py:method:: mindspore.rewrite.PatternNode.match(node: Node)

        检查当前PatternNode是否可以与node匹配。

        参数：
            - **node** (Node) - 要匹配的节点。

        异常：
            - **TypeError** - 如果参数 `node` 不是PattenNode类型。

    .. py:method:: mindspore.rewrite.PatternNode.name()

        获取PattenNode的名称。

    .. py:method:: mindspore.rewrite.PatternNode.set_inputs(inputs)

        设置当前PatternNode的输入。

        参数：
            - **inputs** (list[PatternNode]) - 设置为当前PatternNode的输入。

        异常：
            - **TypeError** - 如果参数 `inputs` 不是list或者 `inputs` 的成员不是PattenNode类型。

    .. py:method:: mindspore.rewrite.PatternNode.type()

        获取PattenNode的类型。


.. py:class:: mindspore.rewrite.VarNode()

    VarNode是PatternNode的子类，其匹配方法始终返回True。

.. py:class:: mindspore.rewrite.Replacement

    替换的接口定义。

    .. py:method:: mindspore.rewrite.Replacement.build(pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict)
        :abstractmethod:

        用于从匹配结果创建替换节点的接口定义。

        .. note::
            返回值将作为SymbolTree的替换函数的参数，返回值应遵循替换函数参数的 `new_nodes` 的约束。请参阅SymbolTree的 `replace` 的文档字符串中的详细信息。

        参数：
            - **pattern** (PatternNode) - 当前模式的根节点。
            - **is_chain_pattern** (bool) - 标记，标记模式是链模式或树模式。
            - **matched** (OrderedDict) - 匹配结果，从名称映射到节点的字典。

        返回：
            作为替换节点的节点实例列表。

.. py:class:: mindspore.rewrite.TreeNodeHelper

    TreeNodeHelper用于在从Tree类型节点获取 `symbol_tree` 时打破循环引用。

    TreeNodeHelper提供了静态方法 `get_sub_tree` 用于从Tree类型节点获取 `symbol_tree`。

    .. py:method:: mindspore.rewrite.TreeNodeHelper.get_sub_tree(node: Node)
        :staticmethod:

        获取Tree类型节点的 `symbol_tree`。

        参数：
            - **node** (Node) - 可以持有子符号树的节点。

        返回：
            Tree节点中的SymbolTree对象。注意节点的 `symbol_tree` 可能是None，在这种情况下，方法将返回None。

        异常：
            - **RuntimeError** - 如果参数 `node` 不是 NodeType.Tree类型。
            - **TypeError** - 如果参数 `node` 不是Node类型实例。

.. py:function:: mindspore.rewrite.sparsify(f, arg_types, sparse_rules=None)

    模型自动稀疏化接口，将稠密模型转换为稀疏模型。通过 `arg_types` 指定的参数类型，将稀疏参数在模型中传导，并调用相应的稀疏函数。

    参数：
        - **f** (Cell) - 被稀疏化的网络。
        - **arg_types** (Tuple[ArgType] | Dict[int, ArgType]) - `f` 接受的参数类型（稀疏CSR/COO、非稀疏等）。如果是tuple，长度需要和 `f` 的参数数量相等；如果是dict，每个键值对应一个参数的索引，字典里没有表示的参数默认为非稀疏。
        - **sparse_rules** (Dict[str, SparseFunc], Optional) - 自定义稀疏规则。默认为None。

.. py:class:: mindspore.rewrite.ArgType

    稀疏化的参数类型。

    - CSR表示CSRTensor
    - COO表示COOTensor
    - NONSPARSE表示非稀疏

.. py:class:: mindspore.rewrite.SparseFunc(fn: Union[str, Callable], inputs: Optional[Any] = None, outputs: Optional[Any] = None)

    在稀疏化中表示一个稀疏函数。

    .. note::
        如果 `fn` 是一个包含类型注解的函数，且同时提供了 `inputs`，则类型注解中的输入类型将被忽略。`outputs` 同理。

    参数：
        - **fn** (Union[str, Callable]) - 稀疏函数，如果是字符串，表示一个mindspore.ops.function接口；或者是任意函数对象。
        - **inputs** (Optional[Any]) - 函数的输入类型。如果是None，则使用函数本身的类型注解。默认是None。
        - **outputs** (Optional[Any]) - 函数的输出类型。如果是None，则使用函数本身的类型注解。默认是None。


