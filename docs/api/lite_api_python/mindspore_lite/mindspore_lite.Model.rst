mindspore_lite.Model
====================

.. py:class:: mindspore_lite.Model()

    `Model` 类定义MindSpore Lite模型，便于计算图管理。

    .. py:method:: build_from_file(model_path, model_type, context=None, config_path="")

        从文件加载并构建模型。

        参数：
            - **model_path** (str) - 定义输入模型文件的路径，例如："/home/user/model.ms"。选项：MindSpore模型: "model.mindir" | MindSpore Lite模型: "model.ms"
            - **model_type** (ModelType) - 定义输入模型文件的类型。选项：ModelType::MINDIR | ModelType::MINDIR_LITE。有关详细信息，请参见 `模型类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.ModelType.html>`_ 。
            - **context** (Context，可选) - 定义上下文，用于在执行期间传递选项。默认值：None。None表示设置target为cpu的Context。
            - **config_path** (str，可选) - 定义配置文件的路径，用于在构建模型期间传递用户定义选项。在以下场景中，用户可能需要设置参数。例如："/home/user/config.txt"。默认值：""。

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
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `model_type` 不是ModelType类型。
            - **TypeError** - `context` 既不是Context类型也不是None。
            - **TypeError** - `config_path` 不是str类型。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - `config_path` 文件路径不存在。
            - **RuntimeError** - 从 `config_path` 加载配置文件失败。
            - **RuntimeError** - 从文件加载并构建模型失败。

    .. py:method:: get_inputs()

        获取模型的所有输入Tensor。

        返回：
            list[Tensor]，模型的输入Tensor列表。

    .. py:method:: predict(inputs)

        推理模型。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入Tensor的顺序列表。

        返回：
            list[Tensor]，模型的输出Tensor列表。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
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
