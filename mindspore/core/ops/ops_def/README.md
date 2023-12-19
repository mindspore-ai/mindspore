### Yaml file rules and explanations are as follows

```yaml
# Defining the function name and Primitive name of operators, use the '_' to separate words. For example, op_name is 'word1_word2', then the function name is 'word1_word2', and the Primitive class name is 'Word1Word2'.
<op_name>:
  # The 'args' is a fixed key of yaml file to define input args of operators.
  <args>:
    # Mandatory. For every arg, key is operators' argument name, and the value are some items, items' key name can be 'dtype', 'prim_init', 'default', 'type_cast','arg_handler'.
    <arg1>:
      # Mandatory. The 'dtype' is a fixed key.
      # Value is one of {int, float, bool, number, tensor, tuple, list, tuple[int], tuple[float], tuple[bool], tuple[number], tuple[tensor], list[int], list[float], list[bool], list[number], list[tensor]}.
      # If value is 'number', arg can be 'int', 'float' or 'bool'.
      <dtype>: <value>

      # Optional. The 'default' is a fixed key.
      # This item means input arg can use default value.
      <default>: <value>

      # Optional. The 'prim_init' is a fixed key. Value can be 'True' or 'False', arg is arg of '__init__' of Primitive if value is 'True'.
      <prim_init>: <value>

      # Optional. The 'type_cast' is a fixed key. This item means can accept unmatchable input by implicit conversion. Value is one of {int, float, bool, number, tensor, tuple, list, tuple[int], tuple[float], tuple[bool], tuple[number], tuple[tensor], list[int], list[float], list[bool], list[number], list[tensor]}
      # Supported type cast now:
      # 1. int, float, bool, number <-> tensor.
      # 2. int, float, bool, number, tensor <-> list/tuple.
      # 3. list <-> tuple.
      <type_cast>: <value>

      # Optional. The 'arg_handler' is a fixed key. Value is a function name used to convert arg. For example, converting kernel size from 2 to (2, 2).
      <arg_handler>: <value>

    <arg2>:
      ...

    <args_signature>: #Optional
      # Optional. The 'rw_write' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_write>: <arg_name>

      # Optional. The 'rw_read' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_read>: <arg_name>

      # Optional. The 'rw_ref' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_ref>: <arg_name>

      # Optional. arg1 and arg2 should has same dtype. arg3 and arg4 should has same dtype.
      <dtype_group>: (<arg_name1>, <arg_name2>, ...), (<arg_name3>, <arg_name4>, ...), ...

    # The 'returns' is a fixed key of yaml file to define output of operators.
    <returns>:
      # Mandatory. For every output, key is operators' output name, and the value is a item, item's key is 'dtype'.
      <output1>:
        # Mandatory. Just refer to key 'dtype' in args.
        <dtype>: <value>

        # Optional. The 'inplace' is a fixed key. Value is input name of operator if the input is a inplace input.
        <inplace>: <value>

      <output2>:
        ...

    # Optional. Rename the function but not use function name from <op_name>.
    <function>:
      # Optional. The 'name' is a fixed key. Value is the new function name to replace <op_name>.
      <name>: <value>

      # Optional. The 'disable' is a fixed key. Value is 'True' or 'False', the function will not be generated if it is 'True'.
      <disable>: <value>

    # Optional. Reaname the primitive class name but not use class name from <op_name>.
    <class>:
      # Optional. The 'name' is a fixed key. Value is the new class name to replace <op_name>.
      <name>: <value>

    # Optional. The 'view' is a fixed key. Value should be set as 'True' if this is a view operator.
    <view>: <value>

    # Optional. The 'dispatch' is a fixed key. The item is used to control whether generate pyboost codes.
    <dispatch>:
      # Optional. The 'enable' is a fixed key. Pyboost codes will be auto generated if value is True.
      <enable>: <value>

      # Optional. The 'device_name' can be set as 'CPU', 'GPU' or 'Ascend' and the value is a function name. If this item eixst, it means pyboost function cannot be
      # auto generated in specified device target and the specified function defined manually will act as pyboost function.
      <device_name>: <value>
```

### operators definitions will be auto generated when build MindSpore package

The auto generated operator definition python files are in path:

'mindspore/python/mindspore/ops/auto_generate/'.

The auto generated operator definition c++ files are in path:

'mindspore/core/ops/auto_generate/'.

The auto generated operator pyboost code files are in path:

1. 'mindspore/ccsrc/kernel/pyboost/auto_generate'.

2. 'mindspore/ccsrc/plugin/device/ascend/kernel/pyboost/auto_generate'.

3. 'mindspore/ccsrc/plugin/device/gpu/kernel/pyboost/auto_generate'.

4. 'mindspore/ccsrc/plugin/device/cpu/kernel/pyboost/auto_generate'.

5. 'mindspore/ccsrc/pipeline/pynative/op_function/auto_generat'.

