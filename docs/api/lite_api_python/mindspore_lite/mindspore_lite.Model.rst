mindspore_lite.Model
====================

.. py:class:: mindspore_lite.Model()

    Model类用于定义MindSpore模型，便于计算图管理。

    .. py:method:: build_from_file(model_path, model_type, context)

        从文件加载并构建模型。

        参数：
            - **model_path** (str) - 定义模型路径。
            - **model_type** (ModelType) - 定义模型文件的类型。选项：ModelType::MINDIR | ModelType::MINDIR_LITE。

              - **ModelType::MINDIR** - MindSpore模型的中间表示。建议的模型文件后缀为".mindir"。
              - **ModelType::MINDIR_LITE** - MindSpore Lite模型的中间表示。建议的模型文件后缀为".ms"。

            - **context** (Context) - 定义用于在执行期间存储选项的上下文。

        异常：
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `model_type` 不是ModelType类型。
            - **TypeError** - `context` 不是Context类型。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - 从文件加载并构建模型失败。

    .. py:method:: get_input_by_tensor_name(tensor_name)

        按名称获取模型的输入张量。

        参数：
            - **tensor_name** (str) - 张量名称。

        返回：
            Tensor，张量名称的输入张量。

        异常：
            - **TypeError** - `tensor_name` 不是str类型。
            - **RuntimeError** - 按名称获取模型输入张量失败。

    .. py:method:: get_inputs()

        获取模型的所有输入张量。

        返回：
            list[Tensor]，模型的输入张量列表。

    .. py:method:: get_output_by_tensor_name(tensor_name)

        按名称获取模型的输出张量。

        参数：
            - **tensor_name** (str) - 张量名称。

        返回：
            Tensor，张量名称的输出张量。

        异常：
            - **TypeError** - `tensor_name` 不是str类型。
            - **RuntimeError** - 按名称获取模型输出张量失败。

    .. py:method:: get_outputs()

        获取模型的所有输出张量。

        返回：
            list[Tensor]，模型的输出张量列表。

    .. py:method:: predict(inputs, outputs)

        推理模型。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入张量的顺序列表。
            - **outputs** (list[Tensor]) - 模型输出按顺序填充到容器中。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `outputs` 不是list类型。
            - **TypeError** - `outputs` 是list类型，但元素不是Tensor类型。
            - **RuntimeError** - 预测推理模型失败。

    .. py:method:: resize(inputs, dims)

        调整输入形状的大小。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入张量的顺序列表。
            - **dims** (list[list[int]]) - 定义输入张量的新形状的列表，应与输入张量的顺序一致。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `dims` 不是list类型。
            - **TypeError** - `dims` 是list类型，但元素不是list类型。
            - **TypeError** - `dims` 是list类型，元素是list类型，但元素的元素不是int类型。
            - **ValueError** -  `inputs` 的size不等于 `dims` 的size。
            - **RuntimeError** - 调整输入形状的大小失败。
