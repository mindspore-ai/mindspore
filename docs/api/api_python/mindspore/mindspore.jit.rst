mindspore.jit
=============

.. py:function:: mindspore.jit(function=None, *, capture_mode="ast", jit_level="O0", dynamic=0, fullgraph=False, backend="", **options)

    将Python函数编译为一张可调用的MindSpore图。

    MindSpore可以在运行时对图进行优化。

    .. note::
        - 不支持在静态图模式下，运行带装饰@jit(capture_mode="bytecode")的函数，此时该装饰@jit(capture_mode="bytecode")视为无效。
        - 不支持在@jit(capture_mode="ast")装饰的函数内部调用带装饰@jit(capture_mode="bytecode")的函数，该装饰@jit(capture_mode="bytecode")视为无效。

    参数：
        - **function** (Callable, 可选) - 要编译成图的Python函数或者Cell。默认值：``None``。

    关键字参数：
        - **capture_mode** (str, 可选) - 创建一张可调用的MindSpore图的方式，可选值有 ``"ast"`` 、 ``"bytecode"`` 和 ``"trace"`` 。默认值： ``"ast"``。

          - ast：解析Python的ast以构建静态图。
          - bytecode：在运行时解析Python字节码以构建静态图。这是一个实验特性，可能会被更改或者删除。暂不支持 Python 3.12 及更高版本。
          - trace：追踪Python代码的执行以构建静态图。这是一个实验特性，可能会被更改或者删除。

        - **jit_level** (str, 可选) - 控制编译优化的级别。目前仅在使用ms_backend后端时生效。可选值有 ``"O0"`` 和 ``"O1"`` 。默认值： ``"O0"``。

          - O0: 除必要影响功能的优化外，其他优化均关闭。
          - O1: 使能常用优化和自动算子融合优化。

        - **dynamic** (int, 可选) - 是否需要进行动态shape编译。默认值： ``0``。其值范围如下：

          - 0: 不进行动态shape编译。
          - 1: 使能动态shape编译，自动检测shape的变化。

        - **fullgraph** (bool, 可选) - 是否捕获整个函数来编译成图。如果为False，jit会尽可能地尝试兼容函数中的所有Python语法。如果为True，则需要整个函数都可以被捕获成图，否则（即有不支持的Python语法），会抛出一个异常。当前只对capture_mode为 ``"ast"`` 或 ``"bytecode"`` 时生效。默认值： ``False``。
        - **backend** (str, 可选) - 使用的编译后端。如果该参数未被设置，框架默认Atlas训练系列产品为 ``"GE"`` 后端 ，默认其他产品包括Atlas A2训练系列产品为 ``"ms_backend"`` 后端。

          - ms_backend：使用MindSpore内置的后端引擎进行硬件相关编译优化与执行，可支持Ascend、GPU、CPU等多硬件形态。
          - GE：使用CANN中名为GraphEngine的图编译执行引擎，进行面向Ascend的模型编译与执行。注意：该后端只支持Ascend硬件静态图模式，仅支持整图下沉或者流水线并行下的子图下沉，且不支持动态shape场景。另外，该后端有额外的编译开销以及不易调试调优。

        - **\*\*options** (dict) - 传给编译后端的选项字典。

          某些配置适用于特定的设备和后端，有关详细信息，请参见下表：

          +---------------------------+---------------------------+-------------------------+
          |  配置项                   |  硬件平台支持             |  编译后端支持           |
          +===========================+===========================+=========================+
          | disable_format_transform  |  GPU                      |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+
          | exec_order                |  Ascend                   |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+
          | ge_options                |  Ascend                   |  GE                     |
          +---------------------------+---------------------------+-------------------------+
          | infer_boost               |  Ascend                   |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+
          | auto_offload              |  Ascend                   |  ms_backend             |
          +---------------------------+---------------------------+-------------------------+

          - **disable_format_transform** (bool, 可选) - 表示是否取消NCHW到NHWC的自动格式转换功能。当fp16的网络性能不如fp32时，可以设置 `disable_format_transform` 为 ``True`` ，以尝试提高训练性能。默认值： ``False`` 。
          - **exec_order** (str, 可选) - 算子执行时的排序方法，GRAPH_MODE(0)下jit_level为O0或者O1时生效。不同的执行顺序会使得网络的执行内存和性能有所差异，当前仅支持两种排序方法：bfs和dfs，默认方法为bfs。

            - bfs：默认的排序方法，广度优先排序，具备较好的通信掩盖效果，执行性能相对较好。
            - dfs：可选择的排序方法，深度优先排序，性能相对bfs执行序较差，但内存占用较少，建议在其他执行序OOM的场景下尝试dfs。

          - **ge_options** (dict) - 设置GE的options配置项，配置项分为 ``global`` 和 ``session`` 两类 。这是一个实验特性，可能会被更改或者删除。
            详细的配置请查询 `options配置说明 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendgraphapi/atlasgeapi_07_0146.html>`_ 。

            - global (dict)：设置global类的选项。
            - session (dict)：设置session类的选项。

          - **infer_boost** (str, 可选) - 用来使能推理模式。默认值为 ``"off"``，表示关闭。取值范围如下：

            - on：开启推理模式，推理性能得到较大提升。
            - off：关闭推理模式，使用前向运算进行推理，性能较差。

          - **auto_offload** (str, 可选) - 用来控制是否开启自动节点卸载，该接口为实验性质接口，默认值为 ``""``，不进行卸载。取值范围如下：

            - weight：自动卸载初始化在Remote侧的权重。
            - activaction"：自动卸载激活值数据。
            - all：自动卸载权重以及激活值。

    返回：
        函数，如果 `function` 不是 ``None``，则返回一个已经将输入 `function` 编译成图的可执行函数；如果 `function` 为 ``None``，则返回一个装饰器。当这个装饰器使用单个 `function` 参数进行调用时，等价于 `function` 不是 ``None`` 的场景。
