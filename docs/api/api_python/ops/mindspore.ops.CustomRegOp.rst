mindspore.ops.CustomRegOp
=========================

.. py:class:: mindspore.ops.CustomRegOp(op_name)

    用于为 :class:`mindspore.ops.Custom` 的 `func` 参数生成算子注册信息的类。注册信息主要指定了 `func` 的输入和输出Tensor所支持的数据类型和数据格式、属性以及target信息。

    参数：
        - **op_name** (str) - 表示kernel名称。无需设置此值，因为 `Custom` 会为该参数自动生成唯一名称。默认值："Custom"。

    .. py:method:: input(index=None, name=None, param_type="required", **kwargs)

        指定 :class:`mindspore.ops.Custom` 的 `func` 参数的输入Tensor信息。每次调用该函数都会产生一个输入Tensor信息，也就是说，如果 `func` 有两个输入Tensor，那么该函数应该被连续调用两次。输入Tensor信息将生成为一个字典：{"index": `index`, "name": `name`, "param_type": `param_type`}。

        参数：
            - **index** (int) - 表示输入的索引，从0开始。0表示第一个输入Tensor，1表示第二个输入Tensor，以此类推。如果该值为None，键"index"将不会出现在输入Tensor信息字典中。默认值：None。
            - **name** (str) - 表示第 `index` 个输入的名称。如果该值为None，键"name"将不会出现在输入Tensor信息字典中。默认值：None。
            - **param_type** (str) - 表示第 `index` 个输入的参数类型，可以是["required", "dynamic", "optional"]之一。如果该值为None，键"param_type"将不会出现在输入Tensor信息字典中。默认值："required"。

              - "required": 表示第 `index` 个输入存在并且只能是单个Tensor。
              - "dynamic": 表示第 `index` 个输入存在且Tensor个数可能为多个，比如AddN算子的输入属于这种情况。
              - "optional": 表示第 `index` 个输入存在且为单个Tensor，或者也可能不存在。

            - **kwargs** (dict) - 表示输入的其他信息，用于扩展。

        异常：
            - **TypeError** - `index` 既不是int也不是None。
            - **TypeError** - `name` 既不是str也不是None。
            - **TypeError** - `param_type` 既不是str也不是None。

    .. py:method:: output(index=None, name=None, param_type="required", **kwargs)

        指定 :class:`mindspore.ops.Custom` 的 `func` 参数的输出Tensor信息。每次调用该函数都会产生一个输出Tensor信息，也就是说，如果 `func` 有两个输出Tensor，那么该函数应该被连续调用两次。输出Tensor信息将生成为一个字典：{"index": `index`, "name": `name`, "param_type": `param_type`}。

        参数：
            - **index** (int) - 表示输出的索引，从0开始。0表示第一个输出Tensor，1表示第二个输出Tensor，以此类推。如果该值为None，键"index"将不会出现在输出Tensor信息字典中。默认值：None。
            - **name** (str) - 表示第 `index` 个输出的名称。如果该值为None，键"name"将不会出现在输出Tensor信息字典中。默认值：None。
            - **param_type** (str) - 表示第 `index` 个输出的参数类型，可以是["required", "dynamic", "optional"]之一。如果该值为None，键"param_type"将不会出现在输出Tensor信息字典中。默认值："required"。

              - "required": 表示第 `index` 个输出存在并且只能是单个Tensor。
              - "dynamic": 表示第 `index` 个输出存在且Tensor个数可能为多个。
              - "optional": 表示第 `index` 个输出存在且为单个Tensor，或者也可能不存在。

            - **kwargs** (dict) - 表示输出的其他信息，用于扩展。

        异常：
            - **TypeError** - `index` 既不是int也不是None。
            - **TypeError** - `name` 既不是str也不是None。
            - **TypeError** - `param_type` 既不是str也不是None。

    .. py:method:: dtype_format(*args)

        指定 :class:`mindspore.ops.Custom` 的 `func` 参数的每个输入Tensor和输出Tensor所支持的数据类型和数据格式。正如上面给出的样例，该函数应在 `input` 和 `output` 函数之后被调用。

        参数：
            - **args** (tuple) - 表示（数据类型、格式）组合的列表，`args` 的长度应该等于输入Tensor和输出Tensor数目的总和。 `args` 中的每一项也是一个tuple，tuple[0]和tuple[1]都是str类型，分别指定了一个Tensor的数据类型和数据格式。 :class:`mindspore.ops.DataType` 提供了很多预定义的（数据类型、格式）组合，例如 `DataType.F16_Default` 表示数据类型是float16，数据格式是默认格式。

        异常：
            - **ValueError** - `args` 的长度不等于输入Tensor和输出Tensor数目的总和。

    .. py:method:: attr(name=None, param_type=None, value_type=None, default_value=None, **kwargs)

        指定 :class:`mindspore.ops.Custom` 的 `func` 参数的属性信息。每次调用该函数都会产生一个属性信息，也就是说，如果 `func` 有两个属性，那么这个函数应该被连续调用两次。属性信息将生成为一个字典：{"name": `name`, "param_type": `param_type`, "value_type": `value_type`, "default_value": `default_value`}。

        参数：
            - **name** (str) - 表示属性的名称。如果该值为None，键"index"将不会出现在属性信息字典中。默认值：None。
            - **param_type** (str) - 表示属性的参数类型，可以是["required", "optional"]之一。如果该值为None，键"param_type"将不会出现在属性信息字典中。默认值：None。

              - "required": 表示必须通过在注册信息中设置默认值的方式或者在调用自定义算子时提供输入值的方式来为此属性提供值。
              - "optional": 表示不强制为此属性提供值。

            - **value_type** (str) - 表示属性的值的类型，可以是["int", "str", "bool", "float", "listInt", "listStr", "listBool", "listFloat"]之一。如果该值为None，键"value_type"将不会出现在属性信息字典中。默认值：None。

              - "int": Python int类型的字符串表示。
              - "str": Python str类型的字符串表示。
              - "bool": Python bool类型的字符串表示。
              - "float": Python float类型的字符串表示。
              - "listInt": Python list of int类型的字符串表示。
              - "listStr": Python list of str类型的字符串表示。
              - "listBool": Python list of bool类型的字符串表示。
              - "listFloat": Python list of float类型的字符串表示。

            - **default_value** (str) - 表示属性的默认值。 `default_value` 和 `value_type` 配合使用。如果属性实际的默认值为1.0，那么 `value_type` 是"float", `default_value` 是"1.0"。如果属性实际的默认值是[1, 2, 3]，那么 `value_type` 是"listInt", `default_value` 是"1,2,3"，其中数值通过','分割。如果该值为None，键"default_value"将不会出现在属性信息字典中。目前用于"akg"、"aicpu"和"tbe"类型的自定义算子。默认值：None。
            - **kwargs** (dict) - 表示属性的其他信息，用于扩展。

        异常：
            - **TypeError** - `name` 既不是str也不是None。
            - **TypeError** - `param_type` 既不是str也不是None。
            - **TypeError** - `value_type` 既不是str也不是None。
            - **TypeError** - `default_value` 既不是str也不是None。

    .. py:method:: target(target=None)

        指定当前注册信息所对应的target。

        参数：
            - **target** (str) - 表示当前注册信息所对应的target，可以是["Ascend", "GPU", "CPU"]之一。 对于同一个 :class:`mindspore.ops.Custom` 的 `func` 参数，其在不同的target上可能支持不同的数据类型和数据格式，使用此参数指定注册信息用于哪个target。如果该值为None，它将在 :class:`mindspore.ops.Custom` 内部被自动推断。默认值：None。

        异常：
            - **TypeError** - `target` 既不是str也不是None。

    .. py:method:: get_op_info()

        将生成的注册信息以字典类型返回。正如上面给出的样例， `CustomRegOp` 实例最后调用该函数。
