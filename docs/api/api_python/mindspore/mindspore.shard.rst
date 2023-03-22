mindspore.shard
===============

.. py:function:: mindspore.shard(fn, in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0)

    指定输入/输出Tensor的分布策略，其余算子的策略推导得到。在PyNative模式下，可以利用此方法指定某个Cell以图模式进行分布式执行。 in_strategy/out_strategy需要为元组类型，
    其中的每一个元素指定对应的输入/输出的Tensor分布策略，可参考： :func:`mindspore.ops.Primitive.shard` 的描述。也可以设置为None，会默认以数据并行执行。
    其余算子的并行策略由输入输出指定的策略推导得到。

    .. note::
        需设置执行模式为PyNative模式，同时设置 `set_auto_parallel_context` 中的并行模式为"auto_parallel"且搜索模式(search mode)为"sharding_propagation"。
        如果输入含有Parameter，其对应的策略应该在 `in_strategy` 里设置。
        如果你想了解更多关于shard的信息，可以参考 `函数式算子切分 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pynative_shard_function_parallel.html>`_ 。

    参数：
        - **fn** (Union[Cell, Function]) - 待通过分布式并行执行的函数，它的参数和返回值类型应该均为Tensor或Parameter。
          如果 `fn` 是Cell类型且含有参数，则 `fn` 必须是一个实例化的对象，否则无法访问到其内部参数。
        - **in_strategy** (tuple) - 指定各输入的切分策略，输入元组的每个元素可以为元组或None，元组即具体指定输入每一维的切分策略，None则会默认以数据并行执行。
        - **out_strategy** (Union[tuple, None]) - 指定各输出的切分策略，用法同 `in_strategy`，目前未使能。默认值：None。
        - **parameter_plan** (Union[dict, None]) - 指定各参数的切分策略，传入字典时，键是str类型的参数名，值是一维整数tuple表示相应的切分策略，
          如果参数名错误或对应参数已经设置了切分策略，该参数的设置会被跳过。默认值：None。
        - **device** (string) - 指定执行设备，可以为["CPU", "GPU", "Ascend"]中任意一个，目前未使能。默认值："Ascend"
        - **level** (int) - 指定搜索切分策略的目标函数，即是最大化计算通信比、最小化内存消耗、最大化执行速度等。可以为[0, 1, 2]中任意一个，默认值：0。目前仅支持最大化计算通信比，其余模式未使能。

    返回：
        Function, 返回一个在自动并行流程下执行的函数。

    异常：
        - **AssertionError** - 如果执行模式不是"PYNATIVE_MODE"。
        - **AssertionError** - 如果并行模式不是"auto_parallel"。
        - **AssertionError** - 如果策略搜索模式不是"sharding_propagation"。
        - **AssertionError** - 如果后端不是"Ascend"或"GPU"。
        - **TypeError** - 如果 `in_strategy` 不是tuple。
        - **TypeError** - 如果 `out_strategy` 不是tuple。
        - **TypeError** - 如果 `parameter_plan` 不是dict或None。
        - **TypeError** - 如果 `parameter_plan` 里的任何一个键值类型不是str。
        - **TypeError** - 如果 `parameter_plen` 里的任何一个值类型不是tuple。
