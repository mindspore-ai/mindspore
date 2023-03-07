mindspore_lite.ModelParallelRunner
==================================

.. py:class:: mindspore_lite.ModelParallelRunner()

    `ModelParallelRunner` 类定义了MindSpore Lite的Runner，它支持模型并行。与 `model` 相比， `model` 不支持并行，但 `ModelParallelRunner` 支持并行。一个Runner包含多个worker，worker为实际执行并行推理的单元。典型场景为当多个客户端向服务器发送推理任务时，服务器执行并行推理，缩短推理时间，然后将理结果返回给客户端。

    .. py:method:: build_from_file(model_path, context=None)

        从模型路径构建模型并行Runner，以便它可以在设备上运行。

        参数：
            - **model_path** (str) - 定义模型路径。
            - **context** (Context，可选) - 定义用于在模型池初始化期间传递上下文和选项的配置。默认值：None。None表示设置target为cpu的Context，Context带有默认的parallel属性。

        异常：
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `context` 既不是Context类型也不是None。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - 初始化模型并行Runner失败。

    .. py:method:: get_inputs()

        获取模型的所有输入Tensor。

        返回：
            list[Tensor]，模型的输入Tensor列表。

    .. py:method:: predict(inputs)

        对模型并行Runner进行推理。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入Tensor的顺序列表。

        返回：
            list[Tensor]，模型的输出Tensor列表。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **RuntimeError** - 预测推理模型失败。
