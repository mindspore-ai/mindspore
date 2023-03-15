mindspore_lite.Context
======================

.. py:class:: mindspore_lite.Context()

    `Context` 类用于在执行期间传递环境变量。

    在运行程序之前，应配置context。如果未配置，将设置cpu为target，并默认自动设置cpu属性。

    Context.parallel定义 `ModelParallelRunner` 类的上下文和配置。

    Context.parallel属性：
        - **workers_num** (int) - workers的数量。一个 `ModelParallelRunner` 包含多个worker，worker为实际执行并行推理的单元。将 `workers_num` 设置为0表示 `workers_num` 将基于计算机性能和核心数自动调整。
        - **config_info** (dict{str: dict{str: str}}) - 传递模型权重文件路径的嵌套映射。例如：{"weight": {"weight_path": "/home/user/weight.cfg"}}。key当前支持["weight"]；value为dict格式，其中的key当前支持["weight_path"]，其中的value为权重的路径，例如 "/home/user/weight.cfg"。
        - **config_path** (str) - 定义配置文件的路径，用于在构建 `ModelParallelRunner` 期间传递用户定义选项。在以下场景中，用户可能需要设置参数。例如："/home/user/config.txt"。

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

    .. py:method:: target
        :property:

        获取Context的目标设备信息列表。

        当前支持的target选项：["cpu"] | ["gpu"] | ["ascend"]。

        .. note::
            - gpu添加到target后，将自动地添加cpu作为备份target。因为当检测到gpu不支持某一算子时，系统将尝试cpu是否支持它。此时，需要切换到具有cpu的上下文。
            - ascend添加到target后，将自动地添加cpu作为备份target。在原始模型的输入format与转换生成的模型的输入format不一致的场景时，在Ascend设备上转换生成的模型中将包含 `Transpose` 节点，该节点目前需要在CPU上执行推理，因此需要切换至带有CPU设备信息的Context中。

        Context.cpu属性：
            - **inter_op_parallel_num** (int) - 设置运行时算子的并行数。 `inter_op_parallel_num` 不能大于 `thread_num` 。将 `inter_op_parallel_num` 设置为0表示 `inter_op_parallel_num` 将基于计算机性能和核心数自动调整。
            - **precision_mode** (str) - 设置混合精度模式。选项： "must_keep_origin_dtype" 和 "force_fp16"。

              - **must_keep_origin_dtype** - 保持原有精度的数据类型。
              - **force_fp16** - 强制设置fp16精度。

            - **thread_num** (int) - 设置运行时的线程数。 `thread_num` 不能小于 `inter_op_parallel_num` 。将 `thread_num` 设置为0表示 `thread_num` 将基于计算机性能和核心数自动调整。
            - **thread_affinity_mode** (int) - 设置运行时的CPU绑核策略模式。支持以下 `thread_affinity_mode` 。

              - **0** - 不绑核。
              - **1** - 绑大核优先。
              - **2** - 绑中核优先。

            - **thread_affinity_core_list** (list[int]) - 设置运行时的CPU绑核策略列表。例如：[0,1]代表指定绑定0号CPU和1号CPU。

        Context.gpu属性：
            - **device_id** (int) - 设置设备id。
            - **group_size** (int) - 集群数量，仅获取，不可设置。
            - **precision_mode** (str) - 设置混合精度模式。选项： "must_keep_origin_dtype" 和 "force_fp16"。

              - **must_keep_origin_dtype** - 保持原有精度的数据类型。
              - **force_fp16** - 强制设置fp16精度。

            - **rank_id** (int) - 当前设备在集群中的ID，固定从0开始编号。仅获取，不可设置。

        Context.ascend属性：
            - **device_id** (int) - 设备id。
            - **precision_mode** (str) - 设置混合精度模式。选项： "force_fp16"、 "allow_fp32_to_fp16"、 "must_keep_origin_dtype" 和 "allow_mix_precision"。

              - **force_fp16** - 强制设置fp16精度。
              - **allow_fp32_to_fp16** - 允许fp32精度转换为fp16精度。
              - **must_keep_origin_dtype** - 保持原有精度的数据类型。
              - **allow_mix_precision** - 允许混合精度。

        返回：
            int，Context的目标设备信息。
