mindspore.ops.Custom
=====================

.. py:class:: mindspore.ops.Custom(func, bprop=None, out_dtype=None, func_type="hybrid", out_shape=None, reg_info=None)

    `Custom` 算子是MindSpore自定义算子的统一接口。用户可以利用该接口自行定义MindSpore内置算子库尚未包含的算子。
    根据输入函数的不同，你可以创建多个自定义算子，并且把它们用在神经网络中。
    关于自定义算子的详细说明和介绍，包括参数的正确书写，见 `自定义算子教程 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/operation/op_custom.html>`_ 。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。

    .. note::
        不同自定义算子的函数类型（func_type）支持的平台类型不同。每种类型支持的平台如下：

        - "hybrid": ["GPU", "CPU"].
        - "akg": ["GPU", "CPU"].
        - "aot": ["GPU", "CPU"，"ASCEND"].
        - "pyfunc": ["CPU"].
        - "julia": ["CPU"].

    参数：
        - **func** (Union[function, str]) - 自定义算子的函数表达。

          - function：如果 `func` 是函数类型，那么 `func` 应该是一个Python函数，它描述了用户定义的操作符的计算逻辑。该函数可以是以下之一：
            
            1. AKG操作符实现函数，可以使用ir builder/tvm compute/hybrid语法。
            2. 纯Python函数。
            3. 使用Hybrid DSL编写的带有装饰器的内核函数。

          - 字符串：如果 `func` 是字符串类型，那么 `str` 应该是包含函数名的文件路径。当 `func_type` 是"aot"或"julia"时，可以使用这种方式。

            1. 对于"aot"：

               a) GPU/CPU（仅Linux）平台

               "aot"意味着提前编译，在这种情况下，Custom直接启动用户定义的"xxx.so"文件作为操作符。用户需要提前将手写的"xxx.cu"/"xxx.cc"文件编译成"xxx.so"，并提供文件路径和函数名。

               - "xxx.so"文件生成：

                 1) GPU平台：给定用户定义的"xxx.cu"文件（例如"{path}/add.cu"），使用nvcc命令进行编译（例如"nvcc --shared -Xcompiler -fPIC -o add.so add.cu"）。

                 2) CPU平台：给定用户定义的"xxx.cc"文件（例如"{path}/add.cc"），使用g++/gcc命令进行编译（例如"g++ --shared -fPIC  -o add.so add.cc"）。

               - 定义"xxx.cc"/"xxx.cu"文件：

                 "aot"是一个跨平台的标识符。"xxx.cc"或"xxx.cu"中定义的函数具有相同的参数。通常，该函数应该像这样：

                 .. code-block::

                     int func(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra)

                 参数：

                 - `nparam(int)` : 输入和输出的总数；假设操作符有2个输入和3个输出，那么 `nparam=5` 。
                 - `params(void **)` : 输入和输出指针的数组指针；输入和输出的指针类型为 `void *` ；假设操作符有2个输入和3个输出，那么第一个输入的指针是 `params[0]` ，第二个输出的指针是 `params[3]` 。
                 - `ndims(int *)` : 输入和输出维度数的数组指针；假设 `params[i]` 是一个1024x1024的张量， `params[j]` 是一个77x83x4的张量，那么 `ndims[i]=2` ， `ndims[j]=3` 。
                 - `shapes(int64_t **)` : 输入和输出形状（ `int64_t *` ）的数组指针；第 `i` 个输入的第 `j` 个维度的大小是 `shapes[i][j]` （其中 `0<=j<ndims[i]` ）；假设 `params[i]` 是一个2x3的张量， `params[j]` 是一个3x3x4的张量，那么 `shapes[i][0]=2` ， `shapes[j][2]=4` 。
                 - `dtypes(const char **)` : 输入和输出类型（ `const char *` ）的数组指针；（例如："float32"、"float16"、"float"、"float64"、"int"、"int8"、"int16"、"int32"、"int64"、"uint"、"uint8"、"uint16"、"uint32"、"uint64"、"bool"）
                 - `stream(void *)` : 流指针，仅在CUDA文件中使用。
                 - `extra(void *)` : 用于进一步扩展。

                 返回值（int）:

                 - 0: 如果这个AOT内核成功执行，MindSpore将继续运行。
                 - 其他值: MindSpore将引发异常并退出。

                 示例：详见 `tests/st/ops/graph_kernel/custom/aot_test_files/` 中的详细信息。

               - 在Custom中使用：

                 .. code-block::

                     Custom(func="{dir_path}/{file_name}:{func_name}", ...)

                 例如：Custom(func="./reorganize.so:CustomReorganize", out_shape=[1], out_dtype=mstype.float32, "aot")

               b) ASCEND平台

               在ASCEND平台使用Custom算子之前，用户首先需要基于Ascend C开发自定义算子并编译。算子开发可参考 `快速上手端到端算子开发 <https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0022.html>`_ ，自定义算子编译可使用工具 `Ascend C自定义算子离线编译 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/operation/ascendc_compile.html>`_。
               在入参 `func` 中传入算子的名字, 以自定义算子实现中命名为 `AddCustom` 为例,存在以下几种使用方式:

               - 算子底层调用TBE： `func="AddCustom"`
               - 算子底层调用AclNN： `func="aclnnAddCustom"`
               - 算子的infer shape通过c++推导： `func="infer_shape.cc:aclnnAddCustom"`,其中 `infer_shape.cc` 为c++实现的shape推导。


            2. 对于"julia"：

               目前，"julia"仅支持CPU（仅限Linux平台）。对于julia，它使用JIT编译器（即时编译器），并且julia支持C API来调用julia代码。自定义功能可以直接将用户定义的"xxx.jl"文件作为一个操作符来启动。用户需要编写一个包含模块和函数的"xxx.jl"文件，并提供该文件的路径以及模块名称和函数名称。

               示例：详情见 `tests/st/ops/graph_kernel/custom/julia_test_files/`

               - 在Custom中使用：

                 .. code-block::

                     Custom(func="{dir_path}/{file_name}:{module_name}:{func_name}",...)

                 例如：Custom(func="./add.jl:Add:add", out_shape=[1], out_dtype=mstype.float32, "julia")
        - **out_shape** (Union[function, list, tuple]) - 自定义算子的输入的形状或者输出形状的推导函数。默认值： ``None`` 。
        - **out_dtype** (Union[function, :class:`mindspore.dtype`, tuple[:class:`mindspore.dtype`]]) - 自定义算子的输入的数据类型或者输出数据类型的推导函数。默认值： ``None`` 。
        - **func_type** (str) - 自定义算子的函数类型，必须是[ ``"hybrid"`` , ``"akg"`` , ``"aot"`` , ``"pyfunc"`` , ``"julia"``]中之一。默认值： ``"hybrid"`` 。
        - **bprop** (function) - 自定义算子的反向函数。默认值： ``None``。
        - **reg_info** (Union[str, dict, list, tuple]) - 自定义算子的算子注册信息。默认值： ``None`` 。

    输入：
        - **input** (Union(tuple, list)) - 输入要计算的Tensor。

    输出：
        Tensor。自定义算子的计算结果。

    异常：
        - **TypeError** - 如果输入 `func` 不合法，或者 `func` 对应的注册信息类型不对。
        - **ValueError** - `func_type` 的值不在列表内。
        - **ValueError** - 算子注册信息不合法，包括支持平台不匹配，算子输入和属性与函数不匹配。