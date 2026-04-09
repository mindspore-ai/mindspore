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
      # If arg_handler is not empty, the type of default value should be the first one of 'arg_handler_map'
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
      # Default: `op_name`.
      <name>: <value>

      # Optional. The 'disable' is a fixed key. Value is 'True' or 'False', the function will not be generated if it is 'True'.
      # Default: False
      <disable>: <value>

    # Optional. Reaname the primitive class name but not use class name from <op_name>.
    <class>:
      # Optional. The 'name' is a fixed key. Value is the new class name to replace <op_name>.
      # Default: Transformed from `op_name`. For example, `op_name` is 'avg_pool', then the class name is 'AvgPool'.
      <name>: <value>
      # Optional. The 'disable' is a fixed key. Value is 'True' or 'False', the primitive definition will not be generated if it is 'True'.
      # Default: False.
      <disable>: <value>

    # Optional. The 'view' is a fixed key. Value should be set as 'True' if this is a view operator.
    # Default: False.
    <view>: <value>

    # Optional. The 'composite' is a fixed key. Indicates whether this operator is implemented as a composite operator,
    # i.e. its semantics are defined by composing multiple smaller operators rather than by a dedicated kernel.
    # Default: False.
    <composite>: <value>

    # Optional. The 'dispatch' is a fixed key. The item is used to control whether generate pyboost codes.
    <dispatch>:
      # Optional. The 'enable' is a fixed key. Pyboost codes will be auto generated if value is True.
      # Default: False.
      <enable>: <value>

      # Optional. The 'device_name' can be set as 'CPU', 'GPU' or 'Ascend' and the value is a function name. If this item eixst, it means pyboost function cannot be
      # auto generated in specified device target and the specified function defined manually will act as pyboost function.
      <device_name>: <value>

    # Optional. The 'gen_opapi' is a fixed key for Ascend ACLNN structured operator generation.
    # It is used for operators whose PyBoost Ascend implementation can be generated by describing
    # output tensor information, extra parameters and ACLNN calling convention in yaml.
    <gen_opapi>:
      # Mandatory for each non-inplace output. The key is an output name defined in 'returns'.
      # The item describes how to infer the output tensor shape and dtype before calling ACLNN.
      <output1>:
        # Mandatory for non-inplace outputs. Value can be an input tensor name or a C++ expression.
        # If value is an input tensor name, it will be converted to '<input>->shape()' automatically.
        # For example:
        #   size: input
        #   size: 'ShapeVector{}'
        #   size: '{input->shape()[0] + 2 * pad - weight->shape()[0] + 1, input->shape()[1], weight->shape()[2]}'
        <size>: <value>

        # Mandatory for non-inplace outputs. Value can be an input tensor name or a C++ expression.
        # If value is an input tensor name, it will be converted to '<input>->data_type()' automatically.
        # For example:
        #   dtype: input
        #   dtype: 'promoteTypes(input->data_type(), other->data_type())'
        <dtype>: <value>

      # Optional. The 'new_params' item defines extra C++ local variables generated before ACLNN dispatch.
      # It is used when ACLNN needs parameters derived from original operator arguments.
      # For example:
      #   approximate_mode: 'npu_gelu_approximate_mode(approximate)'
      <new_params>:
        <param_name>: <cpp_expression>

      # Mandatory. The 'exec' item describes ACLNN interface name and dispatch arguments.
      # If only ACLNN interface name is provided, original operator args will be appended automatically.
      # For non-inplace operators, output args will also be appended automatically.
      # For example:
      #   exec: aclnnDot
      # is equivalent to:
      #   exec: aclnnDot, input, other, out
      # For inplace operators, output should be described by 'returns.<output>.inplace',
      # and 'exec' can omit the inplace output argument if ACLNN updates the input directly.
      <exec>: <aclnn_api_name>[, <arg1>, <arg2>, ...]
```

### Structured Ascend ACLNN operator generation

Structured operators are defined in path:

'mindspore/ops/op_def/yaml/structure'.

This mechanism is mainly used to quickly adapt ACLNN operators whose PyBoost Ascend implementation is simple and can be mapped to one ACLNN call. Instead of writing C++ adaptation code for each operator manually, developers can describe the output shape, output dtype, derived parameters and ACLNN dispatch arguments in yaml. The generator will create the PyBoost Ascend implementation automatically. Structured operators should set 'dispatch.enable' to 'True'.

Example with extra ACLNN parameters:

```yaml
npu_gelu:
  args:
    input:
      dtype: tensor
    approximate:
      dtype: str
      default: "'none'"
  returns:
    out:
      dtype: tensor
  dispatch:
    enable: True
  gen_opapi:
    out:
      size: input
      dtype: input
    new_params:
      approximate_mode: 'npu_gelu_approximate_mode(approximate)'
    exec: aclnnGeluV2, input, approximate_mode, out
```

Example with omitted ACLNN dispatch arguments:

```yaml
vdot:
  args:
    input:
      dtype: tensor
    other:
      dtype: tensor
  returns:
    out:
      dtype: tensor
  dispatch:
    enable: True
  gen_opapi:
    out:
      size: 'ShapeVector{}'
      dtype: input
    exec: aclnnDot
```

Example of an inplace structured operator:

```yaml
inplace_addr:
  args:
    input:
      dtype: tensor
    vec1:
      dtype: tensor
    vec2:
      dtype: tensor
    beta:
      dtype: number
      default: 1
    alpha:
      dtype: number
      default: 1
  returns:
    out:
      dtype: tensor
      inplace: input
  dispatch:
    enable: True
  gen_opapi:
    exec: aclnnInplaceAddr
```

### operators definitions will be auto generated when build MindSpore package

The auto generated operator definition python files are in path:

'mindspore/python/mindspore/ops/auto_generate/'.

The auto generated operator definition c++ files are in path:

'mindspore/ops/op_def/auto_generate' (source files .cc).

'mindspore/ops/include/primitive/auto_generate' (header files .h).

The auto generated operator pyboost code files are in path:

1. 'mindspore/pyboost/auto_generate'.

2. 'mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate'.

3. 'mindspore/ops/kernel/gpu/pyboost/auto_generate'.

4. 'mindspore/ops/kernel/cpu/pyboost/auto_generate'.

5. 'mindspore/ccsrc/pynative/forward/pyboost/auto_generate'.

