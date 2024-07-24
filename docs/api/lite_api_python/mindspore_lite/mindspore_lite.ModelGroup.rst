mindspore_lite.ModelGroup
=========================

.. py:class:: mindspore_lite.ModelGroup(flags=ModelGroupFlag.SHARE_WORKSPACE)

    `ModelGroup` 类定义MindSpore Lite模型分组信息，用于共享工作空间（Workspace）内存或者权重（包括常量和变量）内存以及二者同时共享。

    参数：
        - **flags** (ModelGroupFlag，可选) - 指示 `ModelGroup` 的类型。默认： ``ModelGroupFlag.SHARE_WORKSPACE`` 。

    .. py:method:: add_model(models)

        添加需要共享工作空间内存或权重内存的模型。当前参数 `models` 是 `Model` 对象的元组或列表时，仅支持共享权重内存，其他场景下仅支持工作空间内存共享。

        参数：
            - **models** (union[tuple/list(str), tuple/list(Model)]) - 定义共享内存的一组模型文件或模型对象。

        异常：
            - **TypeError** - `models` 不是list和tuple类型。
            - **TypeError** - `models` 是list或tuple类型，但是成员不全是str或者Model对象。
            - **RuntimeError** - 添加模型分组信息失败。

    .. py:method:: cal_max_size_of_workspace(model_type, context)

        计算 `add_model` 模型的工作内存的最大值。仅在 `ModelGroup` 的类型为 ``ModelGroupFlag.SHARE_WORKSPACE`` 时有效。

        参数：
            - **model_type** (ModelType) - 定义模型的类型。
            - **context** (Context) - 定义模型的上下文。

        异常：
            - **TypeError** - `model_type` 不是 `ModelType` 类型。
            - **TypeError** - `context` 不是 `Context` 类型。
            - **RuntimeError** - 计算工作内存的最大值失败。
