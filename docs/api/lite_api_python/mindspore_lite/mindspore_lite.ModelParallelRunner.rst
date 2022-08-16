mindspore_lite.ModelParallelRunner
==================================

.. py:class:: mindspore_lite.ModelParallelRunner()

    ModelParallelRunner类用于定义MindSpore的模型并行的Runner，方便模型管理。

    .. py:method:: init(model_path, runner_config=None)

        从模型路径构建模型并行runner，以便它可以在设备上运行。

        参数：
            - **model_path** (str) - 定义模型路径。
            - **runner_config** (RunnerConfig，可选) - 定义用于在模型池初始化期间存储选项的配置。默认值：None。

        异常：
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `runner_config` 既不是RunnerConfig类型也不是None。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - 初始化模型并行Runner失败。

    .. py:method:: get_inputs()

        获取模型的所有输入张量。

        返回：
            list[Tensor]，模型的输入张量列表。

    .. py:method:: get_outputs()

        获取模型的所有输出张量。

        返回：
            list[Tensor]，模型的输出张量列表。

    .. py:method:: predict(inputs, outputs)

        推理模型并行Runner。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入张量的顺序列表。
            - **outputs** (list[Tensor]) - 模型输出按顺序填充到容器中。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `output` 不是list类型。
            - **TypeError** - `output` 是list类型，但元素不是Tensor类型。
            - **RuntimeError** - 预测推理模型失败。
