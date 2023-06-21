mindspore.rewrite
=================
MindSpore的ReWrite模块为用户提供了基于自定义规则，对网络的前向计算过程进行修改的能力，如插入、删除和替换语句。

使用教程
--------

ReWrite完整示例请参考
`rewrite_example.py <https://gitee.com/mindspore/mindspore/tree/master/docs/api/api_python/rewrite_example.py>`_ 。
该样例代码的主要功能包括：怎么通过网络创建SymbolTree，并且对SymbolTree中的节点进行插入、删除、替换等操作，
其中还包含了对子网络的修改和通过模式匹配进行节点替换。

功能介绍
^^^^^^^^

ReWrite模块使用SymbolTree记录一个网络的前向计算过程，其中计算过程的每条代码语句会被展开，并以节点的形式存储在SymbolTree中。

ReWrite模块提供了一组新的接口，用户可以使用这组接口为一个网络创建SymbolTree，然后对SymbolTree里的节点进行修改，从而实现对
网络前向计算过程的修改。最后得到修改后的网络代码，或者一个新的网络实例。

创建SymbolTree
^^^^^^^^^^^^^^^

当用户需要使用ReWrite模块对一个网络进行修改时，首先需要基于该网络的实例创建一个SymbolTree，使用的接口
是 :func:`mindspore.rewrite.SymbolTree.create` 。

通过接口 :func:`mindspore.rewrite.SymbolTree.get_code` 可以查看当前SymbolTree里存储的网络代码。

.. code-block:: python

    import mindspore.nn as nn
    from mindspore.rewrite import SymbolTree

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(in_channels=32, out_channels=32, has_bias=False, weight_init="ones")
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.dense(x)
            x = self.relu(x)
            return x

    net = MyNet()
    stree = SymbolTree.create(net)
    print(stree.get_code())

运行结果如下：

.. code-block:: python

    import sys
    sys.path.append('...') # Current working directory
    import mindspore
    from mindspore import nn
    import mindspore.nn as nn

    class MyNetOpt(nn.Cell):

        def __init__(self, obj):
            super().__init__()
            for (key, value) in obj.__dict__.items():
                setattr(self, key, value)

        def construct(self, x):
            x = self.dense(x)
            x = self.relu(x)
            return x

可以看到，通过解析网络 `MyNet` ，SymbolTree里存储的新网络的类名是 `MyNetOpt` ，相较原网络增加了后缀 ``Opt`` 。

同时，init函数的参数和内容均发生了改动，新增参数 `obj` 传入的是原始网络的实例，函数里将原始网络的属性信息拷贝到了新的网络里。

新的网络还将当前工作目录保存到 ``sys.path`` 里，从而保证新网络运行时可以搜索到原网络依赖的模块。

通过接口 :func:`mindspore.rewrite.SymbolTree.print_node_tabulate` 可以看到SymbolTree里存储的节点信息及节点拓扑关系。
该接口依赖tabulate模块，安装指令为： ``pip install tabulate`` 。

.. code-block:: python

    stree.print_node_tabulate()

运行结果如下：

.. code-block::

    ================================================================================
    node type          name     codes              arg providers          target users
    -----------------  -------  -----------------  ---------------------  ----------------------
    NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
    NodeType.CallCell  dense    x = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('relu', 0)]]]
    NodeType.CallCell  relu     x = self.relu(x)   [[0, ('dense', 0)]]    [[0, [('return', 0)]]]
    NodeType.Output    return   return x           [[0, ('relu', 0)]]     []
    ==================================================================================

可以看到，网络的前向计算过程的每一条语句均被转换为一个节点，其中每一个节点的名称是唯一的。
SymbolTree里记录了各个Node间的拓扑关系，即节点的某个输入来自哪个节点的第几个输出，以及节点的某个输出被哪些节点的哪个输入使用。

当前向计算过程中存在复杂语句时，创建SymbolTree的过程会将语句展开，然后再将展开后的每个语句转换为节点。

.. code-block:: python

    import mindspore.nn as nn
    from mindspore.rewrite import SymbolTree

    class MyNet_2(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(in_channels=32, out_channels=32, has_bias=False, weight_init="ones")
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.relu(0.5 * self.dense(x))
            return x

    net = MyNet_2()
    stree = SymbolTree.create(net)
    stree.print_node_tabulate()

运行结果如下：

.. code-block::

    ================================================================================
    node type          name        codes                     arg providers             target users
    -----------------  ----------  ------------------------  ------------------------  --------------------------
    NodeType.Input     input_x     x                         []                        [[0, [('dense', 0)]]]
    NodeType.CallCell  dense       dense = self.dense(x)     [[0, ('input_x', 0)]]     [[0, [('binop_mult', 1)]]]
    NodeType.MathOps   binop_mult  mult_var = (0.5 * dense)  [[1, ('dense', 0)]]       [[0, [('relu', 0)]]]
    NodeType.CallCell  relu        x = self.relu(mult_var)   [[0, ('binop_mult', 0)]]  [[0, [('return', 0)]]]
    NodeType.Output    return      return x                  [[0, ('relu', 0)]]        []
    ==================================================================================

可以看到，前向计算过程中写在同一行的dense操作、乘法操作和relu操作，被展开为三行代码，然后被转换为三个对应节点。

插入节点
^^^^^^^^

当需要在网络的前向计算过程中插入一行新的代码时，可以先使用接口 :func:`mindspore.rewrite.Node.create_call_cell` 创建一个新
的节点，然后使用接口 :func:`mindspore.rewrite.SymbolTree.insert` 将创建的节点插入到SymbolTree内。

.. code-block:: python

    from mindspore.rewrite import SymbolTree, Node, ScopedValue
    net = MyNet()
    stree = SymbolTree.create(net)
    new_relu_cell = nn.ReLU()
    new_node = Node.create_call_cell(cell=new_relu_cell, targets=["x"],
                                     args=[ScopedValue.create_naming_value("x")], name="new_relu")
    dense_node = stree.get_node("dense")
    stree.insert(stree.after(dense_node), new_node)
    stree.print_node_tabulate()

在该样例中，插入节点的流程如下：

1. 首先创建了一个新的节点，使用的Cell是 ``nn.ReLU()`` ，输入输出均为 ``"x"`` ，节点名是 ``"new_relu"`` 。
2. 接着通过 :func:`mindspore.rewrite.SymbolTree.get_node` 方法获取dense节点。
3. 最后通过 :func:`mindspore.rewrite.SymbolTree.insert` 方法将新创建的节点插入到dense节点后面。

运行结果如下：

.. code-block::

    ================================================================================
    node type          name      codes                 arg providers           target users
    -----------------  --------  --------------------  ----------------------  ------------------------
    NodeType.Input     input_x   x                     []                      [[0, [('dense', 0)]]]
    NodeType.CallCell  dense     x = self.dense(x)     [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
    NodeType.CallCell  new_relu  x = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('relu', 0)]]]
    NodeType.CallCell  relu      x = self.relu(x)      [[0, ('new_relu', 0)]]  [[0, [('return', 0)]]]
    NodeType.Output    return    return x              [[0, ('relu', 0)]]      []
    ==================================================================================

可以看到，新的new_relu节点插入到dense节点和relu节点间，节点的拓扑结构随着节点插入自动更新。
其中，新节点对应代码里的 `self.new_relu` 定义在新网络的init函数里，使用传入的 `new_relu_cell` 作为实例。

除了使用 :func:`mindspore.rewrite.SymbolTree.get_node` 方法获取节点来指定插入位置，还可以
通过 :func:`mindspore.rewrite.SymbolTree.nodes` 来遍历节点，并使用 :func:`mindspore.rewrite.SymbolTree.get_instance_type`
基于节点对应实例的类型来获取节点，确定插入位置。

.. code-block:: python

    for node in stree.nodes():
        if node.get_instance_type() == nn.Dense:
            stree.insert(stree.after(node), new_node)

如果希望插入新代码的输出不复用原始网络里的变量，可以在创建节点时使用 :func:`mindspore.rewrite.SymbolTree.unique_name` 得
到一个SymbolTree内不重名的变量名，作为节点的输出。

然后在插入节点前，通过使用 :func:`mindspore.rewrite.Node.set_arg` 修改节点输入变量名，设置哪些节点使用新的节点输出作为输入。

.. code-block:: python

    from mindspore.rewrite import SymbolTree, Node, ScopedValue
    net = MyNet()
    stree = SymbolTree.create(net)
    new_relu_cell = nn.ReLU()
    new_node = Node.create_call_cell(cell=new_relu_cell, targets=[stree.unique_name("x")],
                                    args=[ScopedValue.create_naming_value("x")], name="new_relu")
    dense_node = stree.get_node("dense")
    stree.insert(stree.after(dense_node), new_node)
    old_relu_node = stree.get_node("relu")
    old_relu_node.set_arg(0, new_node.get_targets()[0])
    stree.print_node_tabulate()

在该样例中，创建新节点时 `targets` 参数的值进行了不重名的处理，然后将旧的relu节点的输入改为新节点的输出。

运行结果如下：

.. code-block::

    ================================================================================
    node type          name      codes                   arg providers           target users
    -----------------  --------  ----------------------  ----------------------  ------------------------
    NodeType.Input     input_x   x                       []                      [[0, [('dense', 0)]]]
    NodeType.CallCell  dense     x = self.dense(x)       [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
    NodeType.CallCell  new_relu  x_1 = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('relu', 0)]]]
    NodeType.CallCell  relu      x = self.relu(x_1)      [[0, ('new_relu', 0)]]  [[0, [('return', 0)]]]
    NodeType.Output    return    return x                [[0, ('relu', 0)]]      []
    ==================================================================================

可以看到，新节点的输出变量名是一个不重名的名称 ``x_1`` ，且旧的relu节点使用 ``x_1`` 作为输入。

删除节点
^^^^^^^^

当需要在网络的前向计算过程中删除一行代码时，可以使用接口 :func:`mindspore.rewrite.SymbolTree.erase` 来删除节点。

节点删除后，符号树内剩余节点的拓扑关系会依据删除后的代码情况自动更新。
因此，当待删除的节点的输出被别的节点使用时，节点删除后，需要注意剩余节点的拓扑关系是否符合设计预期。

如果待删除节点的前面存在某个节点的输出名和待删除节点的输出名重名，删除节点后，后续使用该输出名作为输入的节点，自动使用前面那个节点
的输出作为输入。拓扑关系会按照该策略更新。

.. code-block:: python

    from mindspore.rewrite import SymbolTree, Node, ScopedValue
    net = MyNet()
    stree = SymbolTree.create(net)
    relu_node = stree.get_node("relu")
    stree.erase(relu_node)
    stree.print_node_tabulate()

运行结果如下：

.. code-block::

    ================================================================================
    node type          name     codes              arg providers          target users
    -----------------  -------  -----------------  ---------------------  ----------------------
    NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
    NodeType.CallCell  dense    x = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
    NodeType.Output    return   return x           [[0, ('dense', 0)]]    []
    ==================================================================================

可以看到，因为dense结点的输出和relu结点的输出同名，删除relu节点后，返回值使用的是dense节点的输出。

如果待删除节点的前面不存在和待删除节点同名的输出，则需要用户先修改后续使用该输出作为输入的节点，更新参数名，然后再
删除节点，以避免删除节点后发生使用了未定义变量的错误。

.. code-block:: python

    import mindspore.nn as nn
    from mindspore.rewrite import SymbolTree

    class MyNet_3(nn.Cell):
        def __init__(self):
            super().__init__()
            self.dense = nn.Dense(in_channels=32, out_channels=32, has_bias=False, weight_init="ones")
            self.relu = nn.ReLU()

        def construct(self, x):
            y = self.dense(x)
            z = self.relu(y)
            return z

    net = MyNet_3()
    stree = SymbolTree.create(net)
    relu_node = stree.get_node("relu")
    for node in relu_node.get_users():
        node.set_arg(0, relu_node.get_args()[0])
    stree.erase(relu_node)
    stree.print_node_tabulate()

在该样例中，拿到relu节点后，先使用接口 :func:`mindspore.rewrite.Node.get_users` 遍历使用relu节点的输出作为输入的节点，将这些
节点的输入都改为relu节点的输入，然后再删除relu节点。这样的话，后续使用了relu节点输出 ``z`` 的地方就都改为使用relu节点输入 ``y`` 了。

具体的参数名修改策略取决于实际场景需求。

运行结果如下：

.. code-block::

    ================================================================================
    node type          name     codes              arg providers          target users
    -----------------  -------  -----------------  ---------------------  ----------------------
    NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
    NodeType.CallCell  dense    y = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
    NodeType.Output    return   return y           [[0, ('dense', 0)]]    []
    ==================================================================================

可以看到，删除relu节点后，最后一个return节点的值从 ``z`` 被更新为 ``y`` 。

替换节点
^^^^^^^^

当需要在网络的前向计算过程中替换代码时，可以使用接口 :func:`mindspore.rewrite.SymbolTree.replace` 来替换节点。

.. code-block:: python

    from mindspore.rewrite import SymbolTree, Node, ScopedValue
    net = MyNet()
    stree = SymbolTree.create(net)
    new_relu_cell = nn.ReLU()
    new_node = Node.create_call_cell(cell=new_relu_cell, targets=["x"],
                                     args=[ScopedValue.create_naming_value("x")], name="new_relu")
    relu_node = stree.get_node("relu")
    stree.replace(relu_node, [new_node])
    stree.print_node_tabulate()

该样例将原始网络里的relu节点替换为new_relu节点，运行结果如下：

.. code-block::

    ================================================================================
    node type          name      codes                 arg providers           target users
    -----------------  --------  --------------------  ----------------------  ------------------------
    NodeType.Input     input_x   x                     []                      [[0, [('dense', 0)]]]
    NodeType.CallCell  dense     x = self.dense(x)     [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
    NodeType.CallCell  new_relu  x = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('return', 0)]]]
    NodeType.Output    return    return x              [[0, ('new_relu', 0)]]  []
    ==================================================================================

如果替换的新节点的输出和被替换节点的输出名不一致，需要注意维护好替换后的节点间的拓扑关系，即先修改后续使用了被替换节点的输出的节点，
更新这些节点的参数名，然后再进行节点替换操作。

.. code-block:: python

    from mindspore.rewrite import SymbolTree, Node, ScopedValue
    net = MyNet()
    stree = SymbolTree.create(net)
    # Update the parameter names of subsequent nodes
    relu_node = stree.get_node("relu")
    for node in relu_node.get_users():
        node.set_arg(0, "y1")
    # Create two new nodes
    new_relu_cell = nn.ReLU()
    new_node = Node.create_call_cell(cell=new_relu_cell, targets=["y1"],
                                    args=[ScopedValue.create_naming_value("x")], name="new_relu_1")
    new_relu_cell_2 = nn.ReLU()
    new_node_2 = Node.create_call_cell(cell=new_relu_cell_2, targets=["y2"],
                                    args=[ScopedValue.create_naming_value("x")], name="new_relu_2")
    # Replace relu node with two new nodes
    stree.replace(relu_node, [new_node, new_node_2])
    stree.print_node_tabulate()

该用例将relu节点替换为两个新的节点，其中第一个节点的输出 ``y1`` 作为返回值更新return节点。运行结果如下：

.. code-block::

    ================================================================================
    node type          name        codes                    arg providers           target users
    -----------------  ----------  -----------------------  ----------------------  -------------------------------------------
    NodeType.Input     input_x     x                        []                      [[0, [('dense', 0)]]]
    NodeType.CallCell  dense       x = self.dense(x)        [[0, ('input_x', 0)]]   [[0, [('new_relu', 0), ('new_relu_1', 0)]]]
    NodeType.CallCell  new_relu    y1 = self.new_relu(x)    [[0, ('dense', 0)]]     [[0, [('return', 0)]]]
    NodeType.CallCell  new_relu_1  y2 = self.new_relu_1(x)  [[0, ('dense', 0)]]     []
    NodeType.Output    return      return y1                [[0, ('new_relu', 0)]]  []
    ==================================================================================

可以看出，relu节点被成功替换为两个新节点，返回值也被更新为第一个新节点的输出。

返回新网络
^^^^^^^^^^

当对网络修改完毕后，就可以使用接口 :func:`mindspore.rewrite.SymbolTree.get_network` 得到修改后的网络实例了。

.. code-block:: python

    new_net = stree.get_network()
    inputs = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)
    outputs = new_net(inputs)

调用该接口后，Rewrite模块会先在当前工作目录的rewritten_network文件夹下，生成修改后的网络对应的脚本文件，然后使用该脚本文件创建新的网络实例，
原网络的实例作为参数使用。新的网络实例可以直接用于计算和训练。

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

        该接口会解析传入的网络实例，将前向计算过程的每一条源码语句展开，并解析为节点，存储在SymbolTree中。

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

    .. py:method:: mindspore.rewrite.SymbolTree.nodes()

        返回当前SymbolTree里节点的生成器，该接口用于遍历SymbolTree里的节点。

        返回：
            当前SymbolTree中节点的生成器。

    .. py:method:: mindspore.rewrite.SymbolTree.print_node_tabulate()

        打印SymbolTree里节点的拓扑信息，包括节点类型、节点名称、节点对应代码、节点的输入输出关系等。
        信息通过print接口输出到屏幕上。

        .. warning::
            - 这是一个实验性API，后续可能修改或删除。

    .. py:method:: mindspore.rewrite.SymbolTree.replace(old_node: Node, new_nodes: [Node])

        使用 `new_nodes` 列表里的节点来替代旧节点 `old_node` 。

        该接口会将 `new_nodes` 里的节点按顺序插入到SymbolTree中，然后删除旧节点 `old_node` 。

        .. note::
            - 仅支持一对一更换或一对多替换。如果需要多对多替换，请参考PatternEngine。
            - 调用者应维护好 `new_nodes` 里每个节点间的拓扑关系，以及 `new_nodes` 里的节点与原始树中节点的拓扑关系。

        参数：
            - **old_node** (Node) - 被替换节点。
            - **new_nodes** (list[Node]) - 要替换进SymbolTree的节点列表。

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


.. py:class:: mindspore.rewrite.Node(node: NodeImpl)

    节点是表达网络中源码语句的一种数据结构。

    每一个节点通常对应一条前向计算过程展开后的语句。

    节点可以表达前向计算过程的Cell调用语句、Primitive调用语句、算术运算语句、返回语句等。

    参数：
        - **node** (NodeImpl) - `Node` 的内部实现实例。建议调用Node下的指定方法来创建Node，例如 `create_call_cell` ，而不直接
          调用Node的构造函数。不需关心NodeImpl是什么，只需作为句柄看待。

    .. py:method:: mindspore.rewrite.Node.create_call_cell(cell: Cell, targets: [Union[ScopedValue, str]], args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None, name: str = "", is_sub_net: bool = False)
        :staticmethod:

        通过该接口可以根据 `cell` 对象创建一个Node实例。节点对应的源代码格式：

        ``targets = self.name(*args, **kwargs)``。

        参数：
            - **cell** (Cell) - 该节点对应的前向计算的Cell对象。
            - **targets** (list[ScopedValue]) - 表示输出名称。在源代码中作为节点的输出变量名。
            - **args** (list[ScopedValue]) - 该节点的参数名称。用作源代码中代码语句的参数。默认值： ``None`` ，表示 `cell` 没有参数输入。
            - **kwargs** (dict) - 键的类型必须是str，值的类型必须是ScopedValue。用来说明带有关键字的形参的输入参数名称。输入名称在源代码中作为语句表达式中的 `kwargs`。默认值： ``None`` ，表示 `cell` 没有 `kwargs` 输入。
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
    - **CallFunction**： `CallFunction` 节点代表在前向计算中调用MindSpore函数。
    - **CallMethod**： `CallMethod` 不能对应到Cell或者Primitive的节点。
    - **Python**： `Python` 节点包含不支持的 `ast` 的节点类型或不必要的解析 `ast` 节点。
    - **Input**：输入节点代表SymbolTree的输入，对应方法的参数。
    - **Output**: 输出节点代表SymbolTree的输出，对应方法的 `return` 语句。
    - **Tree**: 树节点代表前向计算中调用了别的网络。
    - **MathOps**： 运算符节点代表在前向计算中的一个运算操作，如加法运算或比较运算。

.. py:class:: mindspore.rewrite.ScopedValue(arg_type: ValueType, scope: str = "", value=None)

    ScopedValue表示具有完整范围的值。

    ScopedValue用于表示：左值，如赋值语句的目标，或可调用对象，如调用语句的 `func`，或右值，如赋值语句的 `args` 和 `kwargs`。

    参数：
        - **arg_type** (ValueType) - 当前值的类型。
        - **scope** (str) - 字符串表示当前值的范围。以"self.var1"为例，这个var1的作用域是"self"。默认值： ``""`` 。
        - **value** - 当前ScopedValue中保存的值。值的类型对应于 `arg_type`。默认值： ``None`` 。

    .. py:method:: mindspore.rewrite.ScopedValue.create_name_values(names: Union[list, tuple], scopes: Union[list, tuple] = None)
        :staticmethod:

        创建ScopedValue的列表。

        参数：
            - **names** (list[str] or tuple[str]) - 引用变量的名称，类型为str的列表或元组。
            - **scopes** (list[str] or tuple[str]) - 引用变量的范围，类型为str的列表或元组。默认值： ``None`` ，表示没有指定作用范围。

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

.. py:class:: mindspore.rewrite.ValueType

    ValueType表示ScopedValue的类型。

    - NamingValue表示对另一个变量的引用。
    - CustomObjValue表示自定义类的实例，或类型超出ValueType的基本类型和容器类型范围的对象。

.. py:class:: mindspore.rewrite.PatternEngine(pattern: Union[PatternNode, List], replacement: Replacement = None)

    PatternEngine通过PattenNode修改SymbolTree。

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

    参数：
        - **pattern** (Union[PatternNode, List]) - PatternNode的实例或用于构造 `Pattent` 的Cell类型列表。
        - **replacement** (callable) - 生成新节点的接口实现。默认值： ``None`` 。

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

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

    参数：
        - **pattern_node_name** (str) - 节点名称。
        - **match_type** (Type) - 当前节点的匹配类型。默认值： ``Type[None]`` 。
        - **inputs** (list[PatternNode]) - 当前节点的输入节点。默认值： ``None`` 。

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

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

.. py:class:: mindspore.rewrite.Replacement

    替换的接口定义。

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

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

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

    .. py:method:: mindspore.rewrite.TreeNodeHelper.get_sub_tree(node: Node)
        :staticmethod:

        获取Tree类型节点的 `symbol_tree`。

        参数：
            - **node** (Node) - 可以持有子SymbolTree的节点。

        返回：
            Tree节点中的SymbolTree对象。注意节点的 `symbol_tree` 可能是None，在这种情况下，方法将返回None。

        异常：
            - **RuntimeError** - 如果参数 `node` 不是 NodeType.Tree类型。
            - **TypeError** - 如果参数 `node` 不是Node类型实例。

.. py:function:: mindspore.rewrite.sparsify(f, arg_types, sparse_rules=None)

    模型自动稀疏化接口，将稠密模型转换为稀疏模型。通过 `arg_types` 指定的参数类型，将稀疏参数在模型中传导，并调用相应的稀疏函数。

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

    参数：
        - **f** (Cell) - 被稀疏化的网络。
        - **arg_types** (Tuple[ArgType] | Dict[int, ArgType]) - `f` 接受的参数类型（稀疏CSR/COO、非稀疏等）。如果是tuple，长度需要和 `f` 的参数数量相等；如果是dict，每个键值对应一个参数的索引，字典里没有表示的参数默认为非稀疏。
        - **sparse_rules** (Dict[str, SparseFunc], 可选) - 自定义稀疏规则。默认值： ``None`` 。

.. py:class:: mindspore.rewrite.ArgType

    稀疏化的参数类型。

    - CSR表示CSRTensor
    - COO表示COOTensor
    - NONSPARSE表示非稀疏

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

.. py:class:: mindspore.rewrite.SparseFunc(fn: Union[str, Callable], inputs: Optional[Any] = None, outputs: Optional[Any] = None)

    在稀疏化中表示一个稀疏函数。

    .. note::
        如果 `fn` 是一个包含类型注解的函数，且同时提供了 `inputs`，则类型注解中的输入类型将被忽略。`outputs` 同理。

    .. warning::
        - 这是一组实验性API，后续可能修改或删除。

    参数：
        - **fn** (Union[str, Callable]) - 稀疏函数，如果是字符串，表示一个mindspore.ops.function接口；或者是任意函数对象。
        - **inputs** (Any, 可选) - 函数的输入类型。如果是 ``None`` ，则使用函数本身的类型注解。默认值： ``None`` 。
        - **outputs** (Any, 可选) - 函数的输出类型。如果是 ``None`` ，则使用函数本身的类型注解。默认值： ``None`` 。



