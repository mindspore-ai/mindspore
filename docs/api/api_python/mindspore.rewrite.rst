mindspore.rewrite
=================
MindSpore的ReWrite模块为用户提供了基于自定义规则，对网络的前向计算过程进行修改的能力，如插入、删除和替换语句。

如何快速使用ReWrite，请参考 `使用ReWrite修改网络 <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/rewrite/rewrite_tutorial.html>`_ 。

.. py:class:: mindspore.rewrite.Node(node: NodeImpl)

    节点是表达网络中源码语句的一种数据结构。

    每一个节点通常对应一条前向计算过程展开后的语句。

    节点可以表达前向计算过程的Cell调用语句、Primitive调用语句、算术运算语句、返回语句等。

    参数：
        - **node** (NodeImpl) - `Node` 的内部实现实例。建议调用Node下的指定方法来创建Node，例如 `create_call_cell` ，而不直接
          调用Node的构造函数。不需关心NodeImpl是什么，只需作为句柄看待。

    .. py:method:: mindspore.rewrite.Node.create_call_cell(cell: Cell, targets: List[Union[ScopedValue, str]], args: List[ScopedValue] = None, kwargs: Dict[str, ScopedValue] = None, name: str = "", is_sub_net: bool = False)
        :staticmethod:

        通过该接口可以根据 `cell` 对象创建一个Node实例。节点对应的源代码格式：

        ``targets = self.name(*args, **kwargs)``。

        参数：
            - **cell** (Cell) - 该节点对应的前向计算的Cell对象。
            - **targets** (List[Union[ScopedValue, str]]) - 表示输出名称。在源代码中作为节点的输出变量名。
            - **args** (List[ScopedValue]) - 该节点的参数名称。用作源代码中代码语句的参数。默认值： ``None`` ，表示 `cell` 没有参数输入。
            - **kwargs** (Dict[str, ScopedValue]) - 键的类型必须是str，值的类型必须是ScopedValue。用来说明带有关键字的形参的输入参数名称。输入名称在源代码中作为语句表达式中的 `kwargs`。默认值： ``None`` ，表示 `cell` 没有 `kwargs` 输入。
            - **name** (str) - 表示节点的名称。用作源代码中的字段名称。当未提供名称时，ReWrite将根据 `target` 生成一个默认名称。Rewrite将在插入节点时检查并确保名称的唯一性。默认值： ``""`` 。
            - **is_sub_net** (bool) - 表示 `cell` 是否是一个网络。如果 `is_sub_net` 为 ``True`` ，Rewrite将尝试将 `cell` 解析为TreeNode，否则为CallCell节点。默认值： ``False`` 。

        返回：
            Node实例。

        异常：
            - **TypeError** - 如果参数 `cell` 不是Cell类型。
            - **TypeError** - 如果参数 `targets` 不是list类型。
            - **TypeError** - 如果参数 `targets` 的成员不是str或者ScopedValue类型。
            - **TypeError** - 如果参数 `args` 不是ScopedValue类型。
            - **TypeError** - 如果参数 `kwarg` 的 `key` 不是str类型或者 `value` 不是ScopedValue类型。

    .. py:method:: mindspore.rewrite.Node.create_call_function(function: FunctionType, targets: List[Union[ScopedValue, str]], args: List[ScopedValue] = None, kwargs: Dict[str, ScopedValue] = None)
        :staticmethod:

        通过该接口可以根据一个函数调用创建一个Node实例。 `function` 对象会被保存在网络里，然后通过 `self.` 方法来调用这个函数对象。

        参数：
            - **function** (FunctionType) - 被调用的函数定义。
            - **targets** (List[Union[ScopedValue, str]]) - 表示输出名称。在源代码中作为节点的输出变量名。
            - **args** (List[ScopedValue]) - 该节点的参数名称。用作源代码中代码语句的参数。默认值： ``None`` ，表示 `function` 没有参数输入。
            - **kwargs** (Dict[str, ScopedValue]) - 键的类型必须是str，值的类型必须是ScopedValue。用来说明带有关键字的形参的输入参数名称。输入名称在源代码中作为语句表达式中的 `kwargs`。默认值： ``None`` ，表示 `function` 没有 `kwargs` 输入。

        返回：
            Node实例。

        异常：
            - **TypeError** - 如果参数 `function` 不是FunctionType类型。
            - **TypeError** - 如果参数 `targets` 不是list类型。
            - **TypeError** - 如果参数 `targets` 的成员不是str或者ScopedValue类型。
            - **TypeError** - 如果参数 `args` 不是ScopedValue类型。
            - **TypeError** - 如果参数 `kwarg` 的 `key` 不是str类型或者 `value` 不是ScopedValue类型。

    .. py:method:: mindspore.rewrite.Node.get_args()

        获取当前节点的参数列表。

        返回：
            参数列表，参数类型为 ``ScopedValue`` 。

    .. py:method:: mindspore.rewrite.Node.get_inputs()

        获取一个节点列表，列表里的节点的输出作为当前节点的输入。

        返回：
            节点列表。

    .. py:method:: mindspore.rewrite.Node.get_instance_type()

        获取当前节点对应的代码语句里调用的对象类型。

        - 如果当前节点的 `node_type` 是 `CallCell`，表示该节点的语句调用了一个 ``Cell`` 类型对象。
        - 如果当前节点的 `node_type` 是 `CallPrimitive`，表示该节点的语句调用了一个 ``Primitive`` 类型对象。
        - 如果当前节点的 `node_type` 是 `Tree`，表示该节点的语句调用了一个网络类型的对象。
        - 如果当前节点的 `node_type` 是 `Python`、 `Input`、 `Output`、 `CallMethod`，返回的对象类型是 ``NoneType`` 。

        返回：
            当前节点对应的代码语句里调用的对象类型。

    .. py:method:: mindspore.rewrite.Node.get_name()

        获取当前节点的名称。当节点被插入到SymbolTree时，节点的名称在SymbolTree中应该是唯一的。

        返回：
            节点的名称，类型为str。

    .. py:method:: mindspore.rewrite.Node.get_node_type()

        获取当前节点的类型。节点类型详见 :class:`mindspore.rewrite.NodeType` 。

        返回：
            NodeType，当前节点的类型。

    .. py:method:: mindspore.rewrite.Node.get_symbol_tree()

        获取当前节点所属的SymbolTree。

        返回：
            SymbolTree，如果当前节点不属于任何SymbolTree，则返回 ``None`` .

    .. py:method:: mindspore.rewrite.Node.get_targets()

        获取当前节点的输出值列表。

        返回：
            输出值列表，参数类型为 ``ScopedValue`` 。

    .. py:method:: mindspore.rewrite.Node.get_users()

        获取一个节点列表，列表里的节点使用当前节点的输出作为输入。

        返回：
            节点列表。

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
            - **out_idx** (int，可选) - 指定输入节点的哪个输出作为当前节点输入，则取第一个输出。默认值： ``None`` 。

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
    - **CallFunction**： `CallFunction` 节点代表在前向计算中调用了一个函数。
    - **CallMethod**： `CallMethod` 不能对应到Cell或者Primitive的节点。
    - **Python**： `Python` 节点代表不支持的 `ast` 节点或无需解析的 `ast` 节点。
    - **Input**： `Input` 节点代表SymbolTree的输入，对应方法的参数。
    - **Output**： `Output` 节点代表SymbolTree的输出，对应方法的 `return` 语句。
    - **Tree**： `Tree` 节点代表前向计算中调用了别的网络。
    - **CellContainer**: `CellContainer` 节点代表在前向计算中调用 :class:`mindspore.nn.SequentialCell` 函数。
    - **MathOps**： `MathOps` 节点代表在前向计算中的一个运算操作，如加法运算或比较运算。
    - **ControlFlow**： `ControlFlow` 节点代表一个控制流语句，如 `if` 语句。

.. py:class:: mindspore.rewrite.ScopedValue(arg_type: ValueType, scope: str = "", value=None)

    ScopedValue表示具有完整范围的值。

    ScopedValue用于表示：左值，如赋值语句的目标，或可调用对象，如调用语句的 `func`，或右值，如赋值语句的 `args` 和 `kwargs`。

    参数：
        - **arg_type** (ValueType) - 当前值的类型。
        - **scope** (str) - 字符串表示当前值的范围。以"self.var1"为例，这个var1的作用域是"self"。默认值： ``""`` 。
        - **value** - 当前ScopedValue中保存的值。值的类型对应于 `arg_type`。默认值： ``None`` 。

    .. py:method:: mindspore.rewrite.ScopedValue.create_name_values(names: Union[List[str], Tuple[str]], scopes: Union[List[str], Tuple[str]] = None)
        :staticmethod:

        创建ScopedValue的列表。

        参数：
            - **names** (List[str] or Tuple[str]) - 引用变量的名称，类型为str的列表或元组。
            - **scopes** (List[str] or Tuple[str]) - 引用变量的范围，类型为str的列表或元组。默认值： ``None`` ，表示没有指定作用范围。

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
            - **scope** (str) – 表示变量范围的字符串，默认值： ``""`` ，表示没有指定作用范围。

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

.. py:class:: mindspore.rewrite.SymbolTree(handler: SymbolTreeImpl)

    SymbolTree保存了一个网络的信息，包括网络前向计算过程的语句，和语句输入输出之间的拓扑关系。

    网络里的语句以节点的形式保存在SymbolTree中，通过对SymbolTree里的节点进行处理，可以实现网络代码的删除、插入、替换等操作，
    并得到修改后的网络代码及网络实例。

    参数：
        - **handler** (SymbolTreeImpl) - SymbolTree内部实现实例。建议调用SymbolTree下的 `create` 方法来创建SymbolTree，而不直接
          调用SymbolTree的构造函数。不需关心SymbolTreeImpl是什么，只需作为句柄看待。

    .. py:method:: mindspore.rewrite.SymbolTree.after(node: Union[Node, str])

        返回一个位置信息，位置为 `node` 之后。该接口的返回值作为插入操作的参数使用。

        参数：
            - **node** (Union[Node, str]) - 指定插入位置在哪个节点之后，可以是Node或者Node的名称。

        返回：
            Position，指定插入节点的位置。

        异常：
            - **TypeError** - 参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.before(node: Union[Node, str])

        返回一个位置信息，位置为 `node` 之前。该接口的返回值作为插入操作的参数使用。

        参数：
            - **node** (Union[Node, str]) - 指定插入位置在哪个节点之前，可以是Node或者Node的名称。

        返回：
            Position，指定插入节点的位置。

        异常：
            - **TypeError** - 参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.create(network)

        通过传入网络实例 `network` ，创建一个SymbolTree对象。

        该接口会解析传入的网络实例，将前向计算过程的每一条源码语句展开，并解析为节点，存储在SymbolTree中。具体流程如下：

        1. 获取网络实例对应的源码代码
        2. 对网络进行AST解析，获取网络里各个语句的AST节点（抽象语法树）
        3. 将网络前向计算过程里的复杂语句展开为多个简单语句
        4. 创建SymbolTree对象，每个SymbolTree对应一个网络实例
        5. 使用rewrite节点存储网络前向计算过程的每条语句，节点记录了语句的输入、输出等信息
        6. 将rewrite节点保存到SymbolTree里，同时更新和维护节点间的拓扑连接关系
        7. 返回网络实例对应的SymbolTree对象

        如果网络的前向计算过程里调用了类型为 :class:`mindspore.nn.Cell` 的用户自定义网络，rewrite会为对应语句生成类型
        为 `NodeType.Tree` 的节点，这类节点内部保存了一个新的SymbolTree，这个SymbolTree解析并维护着自定义网络的节点信息。

        如果网络的前向计算过程里调用了以下类型的语句，rewrite会将该语句所对应的内部语句进行解析，并生成对应节点：

        - :class:`mindspore.nn.SequentialCell`
        - 类内函数
        - 控制流语句，如 `if` 语句

        .. note::
            由于网络在rewrite操作期间，控制流的具体执行分支还处于未知状态，因此控制流内部的节点和外部的节点之间不会建立拓扑信息。
            用户在控制流外部使用 :func:`mindspore.rewrite.Node.get_inputs` 和 :func:`mindspore.rewrite.Node.get_users` 接口获取节点时，
            无法获取控制流内部的节点。用户在控制流内部使用这些接口，也无法获取控制流外部的节点。
            因此用户在进行网络修改时，需要手动处理好控制流内部和外部的节点信息。

        当前rewrite模块存在以下语法限制：

        - 仅支持类型为 :class:`mindspore.nn.Cell` 的网络作为rewrite模块的输入。
        - 暂不支持对存在多个输出值的赋值语句进行解析。
        - 暂不支持对循环语句进行解析。
        - 暂不支持对装饰器语法进行解析。
        - 暂不支持对类变量语法进行解析。如果类变量使用了外部数据，可能导致rewrite后的网络出现数据缺失。
        - 暂不支持对局部类和内嵌类进行解析，即类的定义需要放在最外层。
        - 暂不支持对闭包语法进行解析，即类外函数的定义需要放在最外层。
        - 暂不支持对lambda表达式语法进行解析。

        对于不支持解析的语句，rewrite会为对应语句生成类型为 `NodeType.Python` 的节点，以确保rewrite后的网络可以正常运行。
        `Python` 节点不支持对语句的输入和输出进行修改，且可能出现变量名与rewrite生成的变量名的问题，此时用户需要手动对变量名进行调整。

        参数：
            - **network** (Cell) - 待修改的网络实例。

        返回：
            SymbolTree，基于 `network` 创建的SymbolTree。

        异常：
            - **TypeError** - 参数 `network` 不是Cell类型对象。

    .. py:method:: mindspore.rewrite.SymbolTree.erase(node: Union[Node, str])

        删除SymbolTree中的一个节点。

        参数：
            - **node** (Union[Node, str]) - 被删除的节点。可以是Node或者Node的名称。

        返回：
            如果 `node` 属于当前的SymbolTree则返回被删除节点。否则返回None。

        异常：
            - **TypeError** - 参数不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.get_code()

        获取SymbolTree里的网络信息所对应的源码。如果网络已经被修改过，则返回的是修改后的源码。

        返回：
            str，SymbolTree对应的源码字符串。

    .. py:method:: mindspore.rewrite.SymbolTree.get_network()

        获取基于SymbolTree生成的网络对象。源码会保存到文件中，文件保存在当前目录的 `rewritten_network` 文件夹里。

        .. note::
            - rewrite模块对网络的修改基于对原有网络实例的AST树的修改实现，且新的网络实例会从原有网络实例里获取属性信息，
              因此，新网络实例和原有网络实例存在数据关联，原有网络不应该再被使用。
            - 由于新网络和原有网络实例存在数据关联，暂不支持使用rewrite生成的源码文件手动创建网络实例。

        返回：
            根据SymbolTree生成的网络对象。

    .. py:method:: mindspore.rewrite.SymbolTree.get_node(node_name: str)

        获取SymbolTree里名称为 `node_name` 的节点。

        参数：
            - **node_name** (str) - 节点名称。

        返回：
            名称为 `node_name` 的节点。如果SymbolTree里没有名称为 `node_name` 的节点，则返回 ``None`` 。

    .. py:method:: mindspore.rewrite.SymbolTree.insert(position, node: Node)

        在SymbolTree的 `position` 位置插入一个节点。 `position` 通过 `before` 或 `after` 来获得。

        参数：
            - **position** (Position) - 插入位置。
            - **node** (Node) - 要插入的节点。

        返回：
            `Node`，被插入的节点。

        异常：
            - **RuntimeError** - 如果 `position` 指定的不是该SymbolTree内的位置。
            - **TypeError** - 如果参数 `position` 不是Position类型。
            - **TypeError** - 如果参数 `node` 不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.nodes(all_nodes: bool = False)

        返回当前SymbolTree里节点的生成器，该接口用于遍历SymbolTree里的节点。

        参数：
            - **all_nodes** (bool) - 获取所有节点，包括在 `CallFunction` 节点、 `CellContainer` 节点和
              子SymbolTree里面的节点。默认值： ``False`` 。

        返回：
            SymbolTree中节点的生成器。

        异常：
            - **TypeError** - 如果参数 `all_nodes` 不是bool类型。

    .. py:method:: mindspore.rewrite.SymbolTree.print_node_tabulate(all_nodes: bool = False)

        打印SymbolTree里节点的拓扑信息，包括节点类型、节点名称、节点对应代码、节点的输入输出关系等。

        信息通过print接口输出到屏幕上，包括以下信息：

        - **node type** (str)：节点类型，具体类型参考 :class:`mindspore.rewrite.NodeType` 。
        - **name** (str)： 节点名称。
        - **codes** (str)： 节点对应的源代码语句。
        - **arg providers** (Dict[int, Tuple[str, int]])： 格式为 `{[idx, (n, k)]}` ，代表该节点的第 `idx` 个参数是节点 `n` 的第 `k` 个输出提供的。
        - **target users** (Dict[int, List[Tuple[str, int]]])： 格式为 `{[idx, [(n, k)]]}` ，代表该节点的第 `idx` 个输出被用作节点 `n` 的第 `k` 个参数。

        参数：
            - **all_nodes** (bool) - 打印所有节点的信息，包括在 `CallFunction` 节点、 `CellContainer` 节点和
              子SymbolTree里面的节点。默认值： ``False`` 。

        异常：
            - **TypeError** - 如果参数 `all_nodes` 不是bool类型。

    .. py:method:: mindspore.rewrite.SymbolTree.replace(old_node: Node, new_nodes: List[Node])

        使用 `new_nodes` 列表里的节点来替代旧节点 `old_node` 。

        该接口会将 `new_nodes` 里的节点按顺序插入到SymbolTree中，然后删除旧节点 `old_node` 。

        .. note::
            - 仅支持一对一更换或一对多替换。如果需要多对多替换，请参考PatternEngine。
            - 调用者应维护好 `new_nodes` 里每个节点间的拓扑关系，以及 `new_nodes` 里的节点与原始树中节点的拓扑关系。

        参数：
            - **old_node** (Node) - 被替换节点。
            - **new_nodes** (List[Node]) - 要替换进SymbolTree的节点列表。

        返回：
            替换到SymbolTree的节点列表的根节点。

        异常：
            - **RuntimeError** - 如果 `old_node` 仍然被其他节点依赖。
            - **TypeError** - 如果参数 `new_nodes` 不是list，或者列表中的成员不是Node类型。
            - **TypeError** - 如果参数 `old_node` 不是Node类型。

    .. py:method:: mindspore.rewrite.SymbolTree.unique_name(name: str = "output")

        基于给定 `name` ，返回一个SymbolTree内唯一的新的名称。当需要一个不冲突的变量名时，可以使用该接口。

        参数：
            - **name** (str, 可选) - 名称前缀。默认值： ``"output"`` 。

        返回：
            str，一个SymbolTree内唯一的新的名称，名称格式为 `name_n` ，其中 `n` 为数字下标。如果输入 `name` 没有名称冲突，则没有数字下标。

        异常：
            - **TypeError** - 如果参数 `name` 不是str类型。

.. py:class:: mindspore.rewrite.ValueType

    ValueType表示ScopedValue的类型。

    - NamingValue表示对另一个变量的引用。
    - CustomObjValue表示自定义类的实例，或类型超出ValueType的基本类型和容器类型范围的对象。
