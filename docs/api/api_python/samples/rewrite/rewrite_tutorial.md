# 使用ReWrite修改网络

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python/samples/rewrite/rewrite_tutorial.md)

此指南展示了[mindspore.rewrite](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html)模块中API的各种用法。

ReWrite完整示例请参考
 [rewrite_example.py](https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python/rewrite_example.py) 。
该样例代码的主要功能包括：怎么通过网络创建SymbolTree，并且对SymbolTree中的节点进行插入、删除、替换等操作，
其中还包含了对子网络的修改和通过模式匹配进行节点替换。

## 功能介绍

ReWrite模块使用SymbolTree记录一个网络的前向计算过程，其中计算过程的每条代码语句会被展开，并以节点的形式存储在SymbolTree中。

ReWrite模块提供了一组新的接口，用户可以使用这组接口为一个网络创建SymbolTree，然后对SymbolTree里的节点进行修改，从而实现对
网络前向计算过程的修改。最后得到修改后的网络代码，或者一个新的网络实例。

## 创建SymbolTree

当用户需要使用ReWrite模块对一个网络进行修改时，首先需要基于该网络的实例创建一个SymbolTree，使用的接口
是 [mindspore.rewrite.SymbolTree.create](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.create) 。

通过接口 [mindspore.rewrite.SymbolTree.get_code](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_code) 可以查看当前SymbolTree里存储的网络代码。

``` python
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
```

运行结果如下：

``` log
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
```

可以看到，通过解析网络 `MyNet` ，SymbolTree里存储的新网络的类名是 `MyNetOpt` ，相较原网络增加了后缀 ``Opt`` 。

同时，init函数的参数和内容均发生了改动，新增参数 `obj` 传入的是原始网络的实例，函数里将原始网络的属性信息拷贝到了新的网络里。

新的网络还将当前工作目录保存到 ``sys.path`` 里，从而保证新网络运行时可以搜索到原网络依赖的模块。

通过接口 [mindspore.rewrite.SymbolTree.print_node_tabulate](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.print_node_tabulate) 可以看到SymbolTree里存储的节点信息及节点拓扑关系。
该接口依赖tabulate模块，安装指令为： ``pip install tabulate`` 。

``` python
stree.print_node_tabulate()
```

运行结果如下：

``` log
================================================================================
node type          name     codes              arg providers          target users
-----------------  -------  -----------------  ---------------------  ----------------------
NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
NodeType.CallCell  dense    x = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('relu', 0)]]]
NodeType.CallCell  relu     x = self.relu(x)   [[0, ('dense', 0)]]    [[0, [('return', 0)]]]
NodeType.Output    return   return x           [[0, ('relu', 0)]]     []
==================================================================================
```

可以看到，网络的前向计算过程的每一条语句均被转换为一个节点，其中每一个节点的名称是唯一的。
SymbolTree里记录了各个Node间的拓扑关系，即节点的某个输入来自哪个节点的第几个输出，以及节点的某个输出被哪些节点的哪个输入使用。

当前向计算过程中存在复杂语句时，创建SymbolTree的过程会将语句展开，然后再将展开后的每个语句转换为节点。

``` python
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
```

运行结果如下：

``` log
================================================================================
node type          name        codes                     arg providers             target users
-----------------  ----------  ------------------------  ------------------------  --------------------------
NodeType.Input     input_x     x                         []                        [[0, [('dense', 0)]]]
NodeType.CallCell  dense       dense = self.dense(x)     [[0, ('input_x', 0)]]     [[0, [('binop_mult', 1)]]]
NodeType.MathOps   binop_mult  mult_var = (0.5 * dense)  [[1, ('dense', 0)]]       [[0, [('relu', 0)]]]
NodeType.CallCell  relu        x = self.relu(mult_var)   [[0, ('binop_mult', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return      return x                  [[0, ('relu', 0)]]        []
==================================================================================
```

可以看到，前向计算过程中写在同一行的dense操作、乘法操作和relu操作，被展开为三行代码，然后被转换为三个对应节点。

## 插入节点

当需要在网络的前向计算过程中插入一行新的代码时，可以先使用接口 [mindspore.rewrite.Node.create_call_cell](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.create_call_cell) 创建一个新
的节点，然后使用接口 [mindspore.rewrite.SymbolTree.insert](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.insert) 将创建的节点插入到SymbolTree内。

``` python
from mindspore.rewrite import SymbolTree, Node, ScopedValue
net = MyNet()
stree = SymbolTree.create(net)
new_relu_cell = nn.ReLU()
new_node = Node.create_call_cell(cell=new_relu_cell, targets=["x"],
                                 args=[ScopedValue.create_naming_value("x")], name="new_relu")
dense_node = stree.get_node("dense")
stree.insert(stree.after(dense_node), new_node)
stree.print_node_tabulate()
```

在该样例中，插入节点的流程如下：

1. 首先创建了一个新的节点，使用的Cell是 ``nn.ReLU()`` ，输入输出均为 ``"x"`` ，节点名是 ``"new_relu"`` 。
2. 接着通过 [mindspore.rewrite.SymbolTree.get_node](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_node) 方法获取dense节点。
3. 最后通过 [mindspore.rewrite.SymbolTree.insert](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.insert) 方法将新创建的节点插入到dense节点后面。

运行结果如下：

``` log
================================================================================
node type          name      codes                 arg providers           target users
-----------------  --------  --------------------  ----------------------  ------------------------
NodeType.Input     input_x   x                     []                      [[0, [('dense', 0)]]]
NodeType.CallCell  dense     x = self.dense(x)     [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
NodeType.CallCell  new_relu  x = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('relu', 0)]]]
NodeType.CallCell  relu      x = self.relu(x)      [[0, ('new_relu', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return    return x              [[0, ('relu', 0)]]      []
==================================================================================
```

可以看到，新的new_relu节点插入到dense节点和relu节点间，节点的拓扑结构随着节点插入自动更新。
其中，新节点对应代码里的 `self.new_relu` 定义在新网络的init函数里，使用传入的 `new_relu_cell` 作为实例。

除了使用 [mindspore.rewrite.SymbolTree.get_node](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_node) 方法获取节点来指定插入位置，还可以通过 [mindspore.rewrite.SymbolTree.nodes](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.nodes) 来遍历节点，并使用 [mindspore.rewrite.Node.get_instance_type](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.get_instance_type) 基于节点对应实例的类型来获取节点，确定插入位置。

``` python
for node in stree.nodes():
    if node.get_instance_type() == nn.Dense:
        stree.insert(stree.after(node), new_node)
```

如果希望插入新代码的输出不复用原始网络里的变量，可以在创建节点时使用 [mindspore.rewrite.SymbolTree.unique_name](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.unique_name) 得
到一个SymbolTree内不重名的变量名，作为节点的输出。

然后在插入节点前，通过使用 [mindspore.rewrite.Node.set_arg](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.set_arg) 修改节点输入变量名，设置哪些节点使用新的节点输出作为输入。

``` python
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
```

在该样例中，创建新节点时 `targets` 参数的值进行了不重名的处理，然后将旧的relu节点的输入改为新节点的输出。

运行结果如下：

``` log
================================================================================
node type          name      codes                   arg providers           target users
-----------------  --------  ----------------------  ----------------------  ------------------------
NodeType.Input     input_x   x                       []                      [[0, [('dense', 0)]]]
NodeType.CallCell  dense     x = self.dense(x)       [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
NodeType.CallCell  new_relu  x_1 = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('relu', 0)]]]
NodeType.CallCell  relu      x = self.relu(x_1)      [[0, ('new_relu', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return    return x                [[0, ('relu', 0)]]      []
==================================================================================
```

可以看到，新节点的输出变量名是一个不重名的名称 ``x_1`` ，且旧的relu节点使用 ``x_1`` 作为输入。

## 删除节点

当需要在网络的前向计算过程中删除一行代码时，可以使用接口 [mindspore.rewrite.SymbolTree.erase](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.erase)  来删除节点。

节点删除后，符号树内剩余节点的拓扑关系会依据删除后的代码情况自动更新。
因此，当待删除的节点的输出被别的节点使用时，节点删除后，需要注意剩余节点的拓扑关系是否符合设计预期。

如果待删除节点的前面存在某个节点的输出名和待删除节点的输出名重名，删除节点后，后续使用该输出名作为输入的节点，自动使用前面那个节点
的输出作为输入。拓扑关系会按照该策略更新。

``` python
from mindspore.rewrite import SymbolTree, Node, ScopedValue
net = MyNet()
stree = SymbolTree.create(net)
relu_node = stree.get_node("relu")
stree.erase(relu_node)
stree.print_node_tabulate()
```

运行结果如下：

``` log
================================================================================
node type          name     codes              arg providers          target users
-----------------  -------  -----------------  ---------------------  ----------------------
NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
NodeType.CallCell  dense    x = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return   return x           [[0, ('dense', 0)]]    []
==================================================================================
```

可以看到，因为dense结点的输出和relu结点的输出同名，删除relu节点后，返回值使用的是dense节点的输出。

如果待删除节点的前面不存在和待删除节点同名的输出，则需要用户先修改后续使用该输出作为输入的节点，更新参数名，然后再
删除节点，以避免删除节点后发生使用了未定义变量的错误。

``` python
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
```

在该样例中，拿到relu节点后，先使用接口 [mindspore.rewrite.Node.get_users](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.get_users)  遍历使用relu节点的输出作为输入的节点，将这些
节点的输入都改为relu节点的输入，然后再删除relu节点。这样的话，后续使用了relu节点输出 ``z`` 的地方就都改为使用relu节点输入 ``y`` 了。

具体的参数名修改策略取决于实际场景需求。

运行结果如下：

``` log
================================================================================
node type          name     codes              arg providers          target users
-----------------  -------  -----------------  ---------------------  ----------------------
NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
NodeType.CallCell  dense    y = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return   return y           [[0, ('dense', 0)]]    []
==================================================================================
```

可以看到，删除relu节点后，最后一个return节点的值从 ``z`` 被更新为 ``y`` 。

## 替换节点

当需要在网络的前向计算过程中替换代码时，可以使用接口 [mindspore.rewrite.SymbolTree.replace](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.replace)  来替换节点。

``` python
from mindspore.rewrite import SymbolTree, Node, ScopedValue
net = MyNet()
stree = SymbolTree.create(net)
new_relu_cell = nn.ReLU()
new_node = Node.create_call_cell(cell=new_relu_cell, targets=["x"],
                                    args=[ScopedValue.create_naming_value("x")], name="new_relu")
relu_node = stree.get_node("relu")
stree.replace(relu_node, [new_node])
stree.print_node_tabulate()
```

该样例将原始网络里的relu节点替换为new_relu节点，运行结果如下：

``` log
================================================================================
node type          name      codes                 arg providers           target users
-----------------  --------  --------------------  ----------------------  ------------------------
NodeType.Input     input_x   x                     []                      [[0, [('dense', 0)]]]
NodeType.CallCell  dense     x = self.dense(x)     [[0, ('input_x', 0)]]   [[0, [('new_relu', 0)]]]
NodeType.CallCell  new_relu  x = self.new_relu(x)  [[0, ('dense', 0)]]     [[0, [('return', 0)]]]
NodeType.Output    return    return x              [[0, ('new_relu', 0)]]  []
==================================================================================
```

如果替换的新节点的输出和被替换节点的输出名不一致，需要注意维护好替换后的节点间的拓扑关系，即先修改后续使用了被替换节点的输出的节点，
更新这些节点的参数名，然后再进行节点替换操作。

``` python
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
```

该用例将relu节点替换为两个新的节点，其中第一个节点的输出 ``y1`` 作为返回值更新return节点。运行结果如下：

``` log
================================================================================
node type          name        codes                    arg providers           target users
-----------------  ----------  -----------------------  ----------------------  -------------------------------------------
NodeType.Input     input_x     x                        []                      [[0, [('dense', 0)]]]
NodeType.CallCell  dense       x = self.dense(x)        [[0, ('input_x', 0)]]   [[0, [('new_relu', 0), ('new_relu_1', 0)]]]
NodeType.CallCell  new_relu    y1 = self.new_relu(x)    [[0, ('dense', 0)]]     [[0, [('return', 0)]]]
NodeType.CallCell  new_relu_1  y2 = self.new_relu_1(x)  [[0, ('dense', 0)]]     []
NodeType.Output    return      return y1                [[0, ('new_relu', 0)]]  []
==================================================================================
```

可以看出，relu节点被成功替换为两个新节点，返回值也被更新为第一个新节点的输出。

## 返回新网络

当对网络修改完毕后，就可以使用接口 [mindspore.rewrite.SymbolTree.get_network](https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_network)  得到修改后的网络实例了。

``` python
from mindspore import Tensor
from mindspore.common import dtype as mstype
import numpy as np
new_net = stree.get_network()
inputs = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)
outputs = new_net(inputs)
```

调用该接口后，Rewrite模块会先在当前工作目录的rewritten_network文件夹下，生成修改后的网络对应的脚本文件，然后使用该脚本文件创建新的网络实例，
原网络的实例作为参数使用。新的网络实例可以直接用于计算和训练。
