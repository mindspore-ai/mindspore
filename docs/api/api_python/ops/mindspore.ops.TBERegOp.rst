mindspore.ops.TBERegOp
========================

.. py:class:: mindspore.ops.TBERegOp(op_name)

    注册TBE算子信息的类。

    参数：
        - **op_name** (str) - 表示算子名称。

    .. py:method:: async_flag(async_flag=False)

        定义算子的计算效率，用于表示是否支持异步计算。

        参数：
            - **async_flag** (bool) - 表示该算子是否异步的标识。默认值：False。

    .. py:method:: attr(name=None, param_type=None, value_type=None, value=None, default_value=None, **kwargs)

        注册TBE算子的属性信息。

        参数：
            - **name** (str) - 表示算子属性的名称。默认值：None。
            - **param_type** (str) - 表示算子属性的参数类型。默认值：None。
            - **value_type** (str) - 表示算子属性的类型。默认值：None。
            - **value** (str) - 表示算子属性的值。默认值：None。
            - **default_value** (str) - 表示算子属性的默认值。默认值：None。
            - **kwargs** (dict) - 表示算子属性的其他信息。

    .. py:method:: binfile_name(binfile_name)

        设置算子底层的二进制文件名，此动作可选。

        参数：
            - **binfile_name** (str) - 表示算子底层的二进制文件名。

    .. py:method:: compute_cost(compute_cost=10)

        定义算子的计算效率，即tiling模块中成本模型的值。

        参数：
            - **compute_cost** (int) - 表示计算成本的值。默认值：10。

    .. py:method:: dynamic_compile_static(dynamic_compile_static=False)

        表示算子是否支持动静合一。

        参数：
            - **dynamic_compile_static** (bool) - 表示静合一的标识。默认值：False。

    .. py:method:: dynamic_shape(dynamic_shape=False)

        表示算子是否支持动态shape。

        参数：
            - **dynamic_shape** (bool) - 表示是否支持动态shape的标识。默认值：False。

    .. py:method:: input(index=None, name=None, need_compile=None, param_type=None, shape=None, value_depend=None, **kwargs)

        注册TBE算子的输入信息。

        参数：
            - **index** (int) - 表示输入的顺序。默认值：None。
            - **name** (str) - 表示当前输入的名称。默认值：None。
            - **need_compile** (bool) - 表示输入是否需要编译。默认值：None。
            - **param_type** (str) - 表示输入的类型。默认值：None。
            - **shape** (str) - 表示输入的shape。默认值：None。
            - **value_depend** (str) - 表示输入是否为常量值。默认值：None。
            - **kwargs** (dict) - 表示输入的其他信息。

    .. py:method:: input_to_attr_index(input_to_attr_index)

        需要转换为属性的输入的索引。

        参数：
            - **input_to_attr_index** (list) - 索引。默认值：()。

    .. py:method:: unknown_shape_formats(unknown_shape_formats)

        动态Shape场景下算子输入/输出Tensor的数据排布。

        参数：
            - **unknown_shape_formats** (list) - 表示动态Shape场景下算子输入/输出Tensor的数据排布。默认值：()，不支持动态Shape。

    .. py:method:: dynamic_rank_support(dynamic_rank_support)

        定义算子是否支持 DynamicRank（动态维度）。

        参数：
            - **dynamic_rank_support** (bool) - 表示算子是否支持 DynamicRank（动态维度）。
              True：表示支持 DynamicRank，算子支持Shape（-2），用于判断是否进行动态。
              False：表示算子不支持DynamicRank。
              默认值：False。

    .. py:method:: is_dynamic_format(is_dynamic_format=False)

        表示算子是否需要op_select_format函数来动态选择合适的数据格式和数据类型。

        参数：
            - **is_dynamic_format** (bool) - 表示否需要op_select_format函数来动态选择合适的数据格式和数据类型的标识。默认值：False。

    .. py:method:: kernel_name(kernel_name)

        表示算子名称。

        参数：
            - **kernel_name** (str) - 表示算子名称。

    .. py:method:: need_check_supported(need_check_supported=False)

        表示算子是否需要检查支持。

        参数：
            - **need_check_supported** (bool) - 表示是否需要检查支持的标识。默认值：False。

    .. py:method:: op_pattern(pattern=None)

        表示算子支持的行为类型。

        参数：
            - **pattern** (str) - 表示算子支持的行为类型，如"broadcast"、"reduce"等。默认值：None。

    .. py:method:: output(index=None, name=None, need_compile=None, param_type=None, shape=None, **kwargs)

        注册TBE算子的输出信息。

        参数：
            - **index** (int) - 表示输出的顺序。默认值：None。
            - **name** (str) - 表示输出的名称。默认值：None。
            - **need_compile** (bool) - 表示输出是否需要编译。默认值：None。
            - **param_type** (str) - 表示输出的类型。默认值：None。
            - **shape** (str) - 表示输出的shape。默认值：None。
            - **kwargs** (dict) - 表示输出的其他信息。

    .. py:method:: partial_flag(partial_flag=True)

        定义算子的计算效率，用于表示是否支持部分计算。

        参数：
            - **partial_flag** (bool) - 表示是否支持部分计算。默认值：True。

    .. py:method:: real_input_index(real_input_index)

        算子前端输入到后端TBE算子输入的映射。

        参数：
            - **real_input_index** (list) - 真实输入的索引。默认值：()。

    .. py:method:: reshape_type(reshape_type)

        指定算子的补维方式。

        参数：
            - **reshape_type** (str) - 指定算子补维方式的值。例如：输入的shape为 :math:`(2, 3)` ，指定reshape_type="CH"，则补维之后的shape为 :math:`(1, 2, 3, 1)` ，即保留CH轴，NW轴补1。
