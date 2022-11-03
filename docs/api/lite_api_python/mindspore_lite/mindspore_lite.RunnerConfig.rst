mindspore_lite.RunnerConfig
===========================

.. py:class:: mindspore_lite.RunnerConfig(context=None, workers_num=None, config_info=None, config_path="")

    RunnerConfig类定义 `ModelParallelRunner` 类的上下文和配置。

    参数：
        - **context** (Context，可选) - 定义上下文，用于在执行期间传递选项。默认值：None。
        - **workers_num** (int，可选) - workers的数量。一个 `ModelParallelRunner` 包含多个worker，worker为实际执行并行推理的单元。将 `workers_num` 设置为0表示 `workers_num` 将基于计算机性能和核心数自动调整。默认值：None，等同于设置为0。
        - **config_info** (dict{str: dict{str: str}}，可选) - 传递模型权重文件路径的嵌套映射。例如：{"weight": {"weight_path": "/home/user/weight.cfg"}}。默认值：None，等同于设置为{}。key当前支持["weight"]；value为dict格式，其中的key当前支持["weight_path"]，其中的value为权重的路径，例如"/home/user/weight.cfg"。
        - **config_path** (str，可选) - 定义配置文件的路径，用于在构建 `ModelParallelRunner` 期间传递用户定义选项。在以下场景中，用户可能需要设置参数。例如："/home/user/config.txt"。默认值：""。

          - **用法1** - 进行混合精度推理的设置，配置文件内容及说明如下：

            .. code-block::

                [execution_plan]
                [op_name1]=data_type:float16（名字为op_name1的算子设置数据类型为Float16）
                [op_name2]=data_type:float32（名字为op_name2的算子设置数据类型为Float32）

          - **用法2** - 在使用GPU推理时，进行TensorRT设置，配置文件内容及说明如下：

            .. code-block::

                [ms_cache]
                serialize_path=[serialization model path]（序列化模型的存储路径）
                [gpu_context]
                input_shape=input_name:[input_dim]（模型输入维度，用于动态shape）
                dynamic_dims=[min_dim~max_dim]（模型输入的动态维度范围，用于动态shape）
                opt_dims=[opt_dim]（模型最优输入维度，用于动态shape）

    异常：
        - **TypeError** - `context` 既不是Context类型也不是None。
        - **TypeError** - `workers_num` 既不是int类型也不是None。
        - **TypeError** - `config_info` 既不是dict类型也不是None。
        - **TypeError** - `config_info` 是dict类型，但key不是str类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，但value不是dict类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，value是dict类型，但value的key不是str类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，value是dict类型，value的key是str类型，但value的value不是str类型。
        - **ValueError** - `workers_num` 是int类型，但小于0。
        - **TypeError** - `config_path` 不是str类型。
        - **ValueError** - `config_path` 文件路径不存在。
