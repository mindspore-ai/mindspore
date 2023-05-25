mindspore_lite.ModelGroup
=========================

.. py:class:: mindspore_lite.ModelGroup()

    `ModelGroup` 类定义MindSpore Lite模型分组信息，用于共享工作空间（Workspace）内存或者权重（包括常量和变量）内存。

    .. py:method:: add_model(models)

        添加需要共享工作空间内存或权重内存的模型。当前参数 `models` 是 `Model` 对象的元组或列表时，仅支持共享权重内存，其他场景下仅支持工作空间内存共享。

        参数：
            - **models** (union[tuple/list(str), tuple/list(Model)]) - 定义共享内存的一组模型文件或模型对象。

        异常：
            - **TypeError** - `models` 不是list和tuple类型。
            - **TypeError** - `models` 是list或tuple类型，但是成员不全是str或者Model对象。
            - **RuntimeError** - 添加模型分组信息失败。
