mindspore.nn.Cell
==================

.. py:class:: mindspore.nn.Cell(auto_prefix=True, flags=None)

   所有神经网络的基类。

   一个 `Cell` 可以是单一的神经网络单元，如 :class:`mindspore.nn.Conv2d`,  :class:`mindspore.nn.ReLU,  :class:`mindspore.nn.BatchNorm`等，也可以是组成网络的 `Cell` 的结合体。

   .. note:: 一般情况下，自动微分 (AutoDiff) 算法会自动调用梯度函数，但是如果使用反向传播方法 (bprop method)，梯度函数将会被反向传播方法代替。反向传播函数会接收一个包含损失对输出的梯度张量 `dout` 和一个包含前向传播结果的张量 `out` 。反向传播过程需要计算损失对输入的梯度，损失对参数变量的梯度目前暂不支持。反向传播函数必须包含自身参数。

   **参数** ：

      - **auto_prefix** (`Cell`) – 递归地生成作用域。默认值：True。
      - **flags** (`dict`) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。

   **支持平台**：

   ``Ascend`` ``GPU`` ``CPU``

   **样例** :

      .. code-block::

            >>> import mindspore.nn as nn
            >>> import mindspore.ops as ops
            >>> class MyCell(nn.Cell):
            ...    def __init__(self):
            ...        super(MyCell, self).__init__()
            ...        self.relu = ops.ReLU()
            ...
            ...    def construct(self, x):
            ...        return self.relu(x)

   .. py:method:: add_flags(**flags)

        为Cell添加自定义属性。

        在实例化Cell类时，如果入参flags不为空，会调用此方法。

        **参数** ：

            **flags** (`dict`) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。


   .. py:method:: add_flags_recursive(**flags)

        如果Cell含有多个子Cell，此方法会递归得给所有子Cell添加自定义属性。

        **参数** ：

            **flags** (`dict`) - Cell的配置信息，目前用于绑定Cell和数据集。用户也通过该参数自定义Cell属性。默认值：None。


   .. py:method:: cast_inputs(inputs, dst_type)

        将输入转换为指定类型。

        **参数** ：

            **inputs** (`tuple[Tensor]`) - 输入。
            **dst_type** (`mindspore.dtype`) - 指定的数据类型。

        **返回**：

            tuple[Tensor]类型，转换类型后的结果。

   .. py:method:: cast_param(param)

      在PyNative模式下，根据自动混合精度的精度设置转换Cell中参数的类型。

      该接口目前在自动混合精度场景下使用。

      **参数**：

         **param** (`Parameter`) – Parameter类型，需要被转换类型的输入参数。

      **返回**：

         Parameter类型，转换类型后的参数。

   .. py:method:: cells()

      返回当前Cell的子Cell的迭代器。


      **返回**：

         Iteration类型，Cell的子Cell。

   .. py:method:: cells_and_names(cells=None, name_prefix="")

      递归地获取当前Cell及输入 `cells` 的所有子Cell的迭代器，包括Cell的名称及其本身。

      **参数**：

         - **cell** (`str`) – 需要进行迭代的Cell。默认值：None。
         - **name_prefix** (`str`) – 作用域。默认值：''。

      **返回**：

         Iteration类型，当前Cell及输入 `cells` 的所有子Cell和相对应的名称。

      **样例** :

            .. code-block::

                  >>> n = Net()
                  >>> names = []
                  >>> for m in n.cells_and_names():
                  ...    if m[0]:
                  ...       names.append(m[0])

   .. py:method:: check_names()

      检查Cell中的网络参数名称是否重复。


   .. py:method:: compile(*inputs)

      编译Cell。

      **参数**：

         **inputs** (`tuple`) – Cell的输入。

   .. py:method:: compile_and_run(*inputs)

      编译并运行Cell。

      **参数**：

         **inputs** (`tuple`) – Cell的输入。

      **返回**：

         Object类型，执行的结果。

   .. py:method:: construct(*inputs, **kwargs)

      定义要执行的计算逻辑。所有子类都必须重写此方法。

      **返回**：

         Tensor类型，返回计算结果。

   .. py:method:: exec_checkpoint_graph()

      保存checkpoint图。

   .. py:method:: extend_repr()

      设置Cell的扩展表示形式。

      若需要在print时输出个性化的扩展信息，请在您的网络中重新实现此方法。

   .. py:method:: generate_scope()

      为网络中的每个Cell对象生成作用域。

   .. py:method:: get_flags()

      获取该Cell的自定义属性。自定义属性通过 `add_flags` 方法添加。

   .. py:method:: get_func_graph_proto()

      返回图的二进制原型。

   .. py:method:: get_parameters(expand=True)

      返回一个该Cell中parameter的迭代器。

      **参数**：

         **expand** (`bool`) – 如果为True，则递归地获取当前Cell和所有子Cell的parameter。否则，只生成当前Cell的子Cell的parameter。默认值：True。

      **返回**：

         Iteration类型，Cell的parameter。

      **样例** :

            .. code-block::

                  >>> n = Net()
                  >>> parameters = []
                  >>> for item in net.get_parameters():
                  ...    parameters.append(item)

   .. py:method:: get_scope()

      返回Cell的作用域。

      **返回**：

         String类型，网络的作用域。


   .. py:method:: insert_child_to_cell(child_name, child_cell)

      将一个给定名称的子Cell添加到当前Cell。

      **参数**：

         - **child_name** (`str`) – 子Cell名称。
         - **child_cell** (`Cell`) – 要插入的子Cell。

      **异常**：

         - **KeyError** – 如果子Cell的名称不正确或与其他子Cell名称重复。
         - **TypeError** – 如果子Cell的类型不正确。

   .. py:method:: insert_param_to_cell(param_name, param, check_name=True)

      向当前Cell添加参数。

      将指定名称的参数插入Cell。目前在 `mindspore.nn.Cell.__setattr__` 中使用。

      **参数**：

         - **param_name** (`str`) – 参数名称。
         - **param** (`Parameter`) – 要插入到Cell的参数。
         - **check_name** (`bool`) – 是否对`param_name`中的"."进行检查。默认值：True。

      **异常**：

         - **KeyError** – 如果参数名称为空或包含"."。
         - **TypeError** – 如果参数的类型不是Parameter。

   .. py:method:: name_cells()

      递归地获取一个Cell中所有子Cell的迭代器。

      包括Cell名称和Cell本身。

      **返回**：

         Dict[String, Cell]，Cell中的所有子Cell及其名称。

   .. py:method:: parameters_and_names(name_prefix='', expand=True)

        返回Cell中parameter的迭代器。

        包含参数名称和参数本身。

        **参数**:
            name_prefix (str): 作用域。默认值： ''。
            expand (bool):  如果为True，则递归地获取当前Cell和所有子Cell的参数及名称；如果为False，只生成当前Cell的子Cell的参数及名称。默认值：True.

        **返回**:
            迭代器，Cell的名称和Cell本身。

        **样例**:
            >>> n = Net()
            >>> names = []
            >>> for m in n.parameters_and_names():
            ...     if m[0]:
            ...         names.append(m[0])

   .. py:property:: param_prefix

      当前Cell的子Cell的参数名前缀。

   .. py:method:: parameters_dict(recurse=True)

      获取此Cell的parameter字典。

      **参数**：

         **recurse** (`bool`) – 是否递归得包含所有子Cell的parameter。默认值：True。

      **返回**：

         OrderedDict类型，返回参数字典。

   .. py:method:: remove_redundant_parameters()

      删除冗余参数。

      这个接口通常不需要显式调用。

   .. py:method:: set_comm_fusion(fusion_type, recurse=True)

      为Cell中的参数设置融合类型。请参考 :class:`mindspore.Parameter.comm_fusion` 的描述。

      .. note:: 当函数被多次调用时，此属性值将被重写。

      **参数**：

         - **fusion_type** (`int`) – Parameter的`comm_fusion` 属性的设置值。
         - **recurse** (`bool`) – 是否递归地设置子Cell的可训练参数。默认值：True。

   .. py:method:: set_grad(requires_grad=True)

      Cell的梯度设置。在PyNative模式下，该参数指定Cell是否需要梯度。如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。

      **参数**：

         **requires_grad** (`bool`) – 指定网络是否需要梯度，如果为True，PyNative模式下Cell将构建反向网络。默认值：True。

      **返回**：

         Cell类型，Cell本身。

   .. py:method:: set_train(mode=True)

      将Cell设置为训练模式。

      设置当前Cell和所有子Cell的训练模式。对于训练和预测具有不同结构的网络层(如 `BatchNorm`)，将通过这个属性区分分支。如果设置为True，则执行训练分支，否则执行另一个分支。

      **参数**：

         **mode** (`bool`) – 指定模型是否为训练模式。默认值：True。

      **返回**：

         Cell类型，Cell本身。

   .. py:method:: to_float(dst_type)

      在Cell和所有子Cell的输入上添加类型转换，以使用特定的浮点类型运行。

      如果 `dst_type` 是 `mindspore.dtype.float16` ，Cell的所有输入(包括作为常量的input， Parameter， Tensor)都会被转换为float16。请参考 `mindspore.build_train_network` 的源代码中的用法。

      .. note:: 多次调用将产生覆盖。

      **参数**：

         **dst_type** (`mindspore.dtype`) – Cell转换为 `dst_type` 类型运行。 `dst_type` 可以是 `mindspore.dtype.float16` 或者  `mindspore.dtype.float32` 。

      **返回**：

         Cell类型，Cell本身。

      **异常**：

         **ValueError** – 如果 `dst_type` 不是 `mindspore.dtype.float32` ，也不是`mindspore.dtype.float16`。

   .. py:method:: trainable_params(recurse=True)

      返回Cell的可训练参数。

      返回一个可训练参数的列表。

      **参数**：

         **recurse** (`bool`) – 是否递归地包含当前Cell的所有子Cell的可训练参数。默认值：True。

      **返回**：

         List类型，可训练参数列表。

   .. py:method:: untrainable_params(recurse=True)

      返回Cell的不可训练参数。

      返回一个不可训练参数的列表。

      **参数**：

         **recurse** (`bool`) – 是否递归地包含当前Cell的所有子Cell的不可训练参数。默认值：True。

      **返回**：

         List类型，不可训练参数列表。

   .. py:method:: update_cell_prefix()

      递归地更新所有子Cell的 `param_prefix` 。

      在调用此方法后，可以通过Cell的 `param_prefix` 属性获取该Cell的所有子Cell的名称前缀。

   .. py:method:: update_cell_type(cell_type)

      量化感知训练网络场景下，更新当前Cell的类型。

      此方法将Cell类型设置为 `cell_type` 。

      **参数**：

         **cell_type** (str) – 被更新的类型，`cell_type` 可以是"quant"或"second-order"。

   .. py:method:: update_parameters_name(prefix="", recurse=True)

      给网络参数名称添加 `prefix` 前缀字符串。

      **参数**：

         - **prefix** (`str`) – 前缀字符串。默认值：''。
         - **recurse** (`bool`) – 是否递归地包含所有子Cell的参数。默认值：True。
