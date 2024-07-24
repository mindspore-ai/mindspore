mindspore_lite.Model
====================

.. py:class:: mindspore_lite.Model()

    `Model` 类定义MindSpore Lite模型，便于计算图管理。

    .. py:method:: build_from_file(model_path, model_type, context=None, config_path="", config_dict: dict = None)

        从文件加载并构建模型。

        参数：
            - **model_path** (str) - 定义输入模型文件的路径，例如："/home/user/model.mindir"。模型应该使用.mindir作为后缀。
            - **model_type** (ModelType) - 定义输入模型文件的类型。选项有 ``ModelType::MINDIR`` 。有关详细信息，请参见 `模型类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.ModelType.html>`_ 。
            - **context** (Context，可选) - 定义上下文，用于在执行期间传递选项。默认值： ``None`` 。 ``None`` 表示设置target为cpu的Context。
            - **config_path** (str，可选) - 定义配置文件的路径，用于在构建模型期间传递用户定义选项。在以下场景中，用户可能需要设置参数。例如："/home/user/config.txt"。默认值： ``""`` 。

              - **用法1** - 进行混合精度推理的设置，配置文件内容及说明如下：

                .. code-block::

                    [execution_plan]
                    [op_name1]=data_type:float16（名字为op_name1的算子设置数据类型为float16）
                    [op_name2]=data_type:float32（名字为op_name2的算子设置数据类型为float32）

              - **用法2** - 在使用GPU推理时，进行TensorRT设置，配置文件内容及说明如下：

                .. code-block::

                    [ms_cache]
                    serialize_path=[serialization model path]（序列化模型的存储路径）
                    [gpu_context]
                    input_shape=input_name:[input_dim]（模型输入维度，用于动态shape）
                    dynamic_dims=[min_dim~max_dim]（模型输入的动态维度范围，用于动态shape）
                    opt_dims=[opt_dim]（模型最优输入维度，用于动态shape）

            - **config_dict** (dict，可选) - 配置参数字典，当使用该字典配置参数时，优先级高于配置文件。

              推理配置rank table。配置文件中的内容及说明如下：

              .. code-block::

                  [ascend_context]
                  rank_table_file=[path_a]（使用路径a的rank table）

              同时配置参数字典中如下：

              .. code-block::

                  config_dict = {"ascend_context" : {"rank_table_file" : "path_b"}}

              那么配置参数字典中路径b的rank table将覆盖配置文件中的路径a的rank table。

        异常：
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `model_type` 不是ModelType类型。
            - **TypeError** - `context` 既不是Context类型也不是 ``None`` 。
            - **TypeError** - `config_path` 不是str类型。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - `config_path` 文件路径不存在。
            - **RuntimeError** - 从 `config_path` 加载配置文件失败。
            - **RuntimeError** - 从文件加载并构建模型失败。

    .. py:method:: get_inputs()

        获取模型的所有输入Tensor。

        返回：
            list[Tensor]，模型的输入Tensor列表。

    .. py:method:: get_outputs()

        获取模型的所有输出Tensor信息。

        返回：
            list[TensorMeta]，模型的输出TensorMeta列表。

    .. py:method:: predict(inputs, outputs=None)

        推理模型。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入Tensor的顺序列表。
            - **outputs** (list[Tensor]，可选) - 包含所有输出Tensor的顺序列表。

        返回：
            list[Tensor]，模型的输出Tensor列表。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `outputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `outputs` 是list类型，但元素不是Tensor类型。
            - **RuntimeError** - 预测推理模型失败。

    .. py:method:: resize(inputs, dims)

        调整输入形状的大小。此方法用于以下场景：

        1. 如果需要预测相同大小的多个输入，可以将 `dims` 的batch（N）维度设置为输入的数量，那么可以同时执行多个输入的推理。

        2. 将输入大小调整为指定shape。

        3. 当输入是动态shape时（模型输入的shape的维度包含-1），必须通过 `resize` 把-1换成固定维度。

        4. 模型中包含的shape算子是动态shape（shape算子的维度包含-1）。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入Tensor的顺序列表。
            - **dims** (list[list[int]]) - 定义输入Tensor的新形状的列表，应与输入Tensor的顺序一致。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `dims` 不是list类型。
            - **TypeError** - `dims` 是list类型，但元素不是list类型。
            - **TypeError** - `dims` 是list类型，元素是list类型，但元素的元素不是int类型。
            - **ValueError** -  `inputs` 的size不等于 `dims` 的size。
            - **RuntimeError** - 调整输入形状的大小失败。

    .. py:method:: get_model_info(key)

        获取模型信息。

        参数：
            - **key** (str) - 获取模型信息关键字，当前支持user_info、input_shape、dynamic_dims、user_info为用户信息，input_shape为模型输入shape，dynamic_dims为动态分档模型支持的尺寸。
        
        异常：
            - key不是str类型。

    .. py:method:: update_weights(weights)

        对模型中的常量Tensor进行权重更新。

        参数：
            - **weights** (list[list[Tensor]]) - 需要更新的Tensor。

        异常：
            - **RuntimeError** - `weights` 不是两层list。
            - **RuntimeError** - `weights` 是list，但是两层list中的元素不是Tensor。
            - **RuntimeError** - 权重更新失败。

        教程样例：
            - `动态权重更新
              <https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_python.html#%E5%8A%A8%E6%80%81%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0>`_
