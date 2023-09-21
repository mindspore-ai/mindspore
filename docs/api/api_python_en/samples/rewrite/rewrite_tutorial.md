# Modifying Network With ReWrite

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python_en/samples/rewrite/rewrite_tutorial.md)

This example illustrates the various usages of APIs available in the [mindspore.rewrite](https://www.mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html) module.

For a complete ReWrite example, refer to
[rewrite_example.py](https://gitee.com/mindspore/mindspore/blob/master/docs/api/api_python_en/rewrite_example.py) .
The main functions of the sample code include: how to create a SymbolTree through the network, and how to insert, delete,
and replace the nodes in the SymbolTree. It also includes the modification of the subnet and node replacement through pattern
matching.

## Function Introduction

ReWrite module uses SymbolTree to record the forward computation of a network, where each code statement of the
forward computation process is expanded and stored in the SymbolTree as nodes.

The ReWrite module provides a new set of interfaces that users can use to create a SymbolTree for a network and then
modify the nodes in the SymbolTree to achieve the network forward computation process modification. Finally, a modified
network code, or a new network instance can be obtained.

## Creating A SymbolTree

When we need to modify a network using the ReWrite module, we first need to create a SymbolTree based on the instance
of the network, using the interface [mindspore.rewrite.SymbolTree.create](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.create) .

Through the use of the interface [mindspore.rewrite.SymbolTree.get_code](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_code), we can view the network code currently
stored in SymbolTree.

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

The results are as follows:

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

It can be seen that by parsing the network `MyNet` , the class name of the new network stored in SymbolTree is `MyNetOpt` ,
which adds the suffix ``Opt`` to the original network.

At the same time, the parameters and content of the init function have been changed. The new parameter `obj` is passed into
the instance of the original network, and the attribute information of the original network is copied to the new network in
the function.

The new network also saves the current working directory to ``sys.path`` , ensuring that modules that the original network
depends on can be searched for when running on the new network.

By using the interface [mindspore.rewrite.SymbolTree.print_node_tabulate](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.print_node_tabulate) , we can see the node information and node
topology relationships stored in the SymbolTree.
This interface depends on the tabulate module, and the installation command is: ``pip install tabulate`` .

``` python
stree.print_node_tabulate()
```

The results are as follows:

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

It can be seen that each statement in the network's forward computation process is converted to a node, where the name
of each node is unique.
The SymbolTree records the topological relationship between each node, that is, the output of which node an input comes
from, and the output of a node is used by which input of which node.

When there are complex statements in the forward computation process, the statements are expanded during the creation
of SymbolTree, and then each expanded statement is converted to a node.

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

The results are as follows:

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

It can be seen that the dense, multiplication, and relu operations written on the same line during forward computing are
expanded into three lines of code and then converted into three corresponding nodes.

## Inserting Nodes

When we need to insert a new line of code during the forward computation of the network, we can first create a new node
using interface [mindspore.rewrite.Node.create_call_cell](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.create_call_cell) , and then insert the created node into SymbolTree
using interface [mindspore.rewrite.SymbolTree.insert](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.insert) .

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

In this example, the process for inserting a node is as follows:

1. Firstly, a new node is created. The Cell used is ``nn.ReLU()`` , the input and output are ``"x"`` , and the node name is ``"new_relu"`` .
2. Then the dense node is fetched by using [mindspore.rewrite.SymbolTree.get_node](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_node) .
3. Finally, the newly created node is inserted after the dense node through [mindspore.rewrite.SymbolTree.insert](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.insert) .

The results are as follows:

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

It can be seen that the new new_relu node is inserted between the dense node and the relu node, and the topology of
node is automatically updated with the node insertion.
The definition of `self.new_relu` in the code of new node is saved in the init function of the new network, using
parameter `new_relu_cell` as the instance.

In addition to getting nodes using [mindspore.rewrite.SymbolTree.get_node](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_node) to specify the insertion location, we can
also iterate through nodes by [mindspore.rewrite.SymbolTree.nodes](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.nodes) and use [mindspore.rewrite.Node.get_instance_type](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.get_instance_type)
to get the node and determine the insertion position based on the type of corresponding instance of node.

``` python
for node in stree.nodes():
    if node.get_instance_type() == nn.Dense:
        stree.insert(stree.after(node), new_node)
```

If we want the output of new code to be inserted does not reuse variables from the original network, we can
use [mindspore.rewrite.SymbolTree.unique_name](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.unique_name) to get an variable name that are not duplicated in the SymbolTree
as the output of node when creating nodes.

Then, before inserting the node, we can modify the node input variable name by using [mindspore.rewrite.Node.set_arg](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.set_arg)
to set which nodes use the new node output as input.

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

In this example, when creating a new node, the value of the `targets` parameter is treated without duplication,
and the input of old relu node is changed to the output of new node.

The results are as follows:

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

It can be seen that the output variable name of new node is an unnamed name ``x_1`` , and the old relu node uses ``x_1`` as input.

## Deleting Nodes

When we need to delete a line of code during the forward computation of the network, we can use the interface
[mindspore.rewrite.SymbolTree.erase](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.erase) to delete the node.

After the node is deleted, the topological relationship of the remaining nodes in the symbol tree will be automatically
updated according to the code of network after deletion.
Therefore, when the output of node to be deleted is used by other nodes, we need to pay attention to whether the topological
relationship of the remaining nodes meets the design expectations after the node is deleted.

If a node exists in front of the node to be deleted that has the same output name as the node to be deleted, after the node
is deleted, the output of the previous node is automatically used as input for the node that uses the output name as the input.
The topology relationship is updated according to this policy.

``` python
from mindspore.rewrite import SymbolTree, Node, ScopedValue
net = MyNet()
stree = SymbolTree.create(net)
relu_node = stree.get_node("relu")
stree.erase(relu_node)
stree.print_node_tabulate()
```

The results are as follows:

``` log
================================================================================
node type          name     codes              arg providers          target users
-----------------  -------  -----------------  ---------------------  ----------------------
NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
NodeType.CallCell  dense    x = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return   return x           [[0, ('dense', 0)]]    []
==================================================================================
```

It can be seen that because the output of dense node and the output of relu node have the same name, after deleting
the relu node, the return value uses the output of the dense node.

If there is no node that has the same output name as the node to be deleted in front of the node to be deleted, we need
to modify subsequent nodes that uses this output as input by updating the input names, and then delete the node, in order
to avoid errors using undefined variables after deleting the node.

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

In this example, after getting the relu node, first we use the interface [mindspore.rewrite.Node.get_users](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.Node.get_users) to
iterate through the nodes that use the output of relu node as input, change the input of these nodes to the input of relu
node, and then delete the relu node. In this case, the subsequent use of the relu node output ``z`` will be changed to
the relu node input ``y`` .

The specific parameter name modification strategy depends on the actual scenario requirements.

The results are as follows:

``` log
================================================================================
node type          name     codes              arg providers          target users
-----------------  -------  -----------------  ---------------------  ----------------------
NodeType.Input     input_x  x                  []                     [[0, [('dense', 0)]]]
NodeType.CallCell  dense    y = self.dense(x)  [[0, ('input_x', 0)]]  [[0, [('return', 0)]]]
NodeType.Output    return   return y           [[0, ('dense', 0)]]    []
==================================================================================
```

It can be seen that after deleting the relu node, the value of the last return node is updated from ``z`` to ``y`` .

## Replacing Nodes

When we need to replace code during the forward computation of network, we can replace the node with the
interface [mindspore.rewrite.SymbolTree.replace](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.replace) .

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

This example replaces relu node in the original network with new_relu node. The results are as follows:

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

If the output name of the new node and the replaced node are inconsistent, we need to pay attention
to maintaining the topological relationship between nodes after replacement, that is, first modify the subsequent nodes that
uses the output of the replaced node, update the parameter names of these nodes, and then perform the node replacement operation.

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

The example replaces relu node with two new nodes, where the output of first node ``y1`` is used as the return value in the
return node. The results are as follows:

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

It can be seen that the relu node was successfully replaced with two new nodes, and the return value was also
updated to the output of the first new node.

## Returning A New Network

When the network is modified, we can use the interface [mindspore.rewrite.SymbolTree.get_network](https://mindspore.cn/docs/en/master/api_python/mindspore.rewrite.html#mindspore.rewrite.SymbolTree.get_network) to get the
modified network instance.

``` python
from mindspore import Tensor
from mindspore.common import dtype as mstype
import numpy as np
new_net = stree.get_network()
inputs = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)
outputs = new_net(inputs)
```

After calling this interface, rewrite module will first generate a script file corresponding to the modified network in the
rewritten_network folder of the current working directory, and then use the script file to create a new network instance,
and use the original network instance as a parameter. New network instances can be used directly for compute and training.
