mindspore_lite.ModelType
========================

.. py:class:: mindspore_lite.ModelType

    `ModelType` 类定义MindSpore Lite中导出或导入的模型类型。

    适用于以下场景：

    1. 调用 `mindspore_lite.Converter` 时，设置 `save_type` 参数， `ModelType` 用于定义转换生成的模型类型。

    2. 调用 `mindspore_lite.Converter` 之后，当从文件加载或构建模型以进行推理时， `ModelType` 用于定义输入模型框架类型。

    目前，支持以下 `ModelType` ：

    ===========================  ================================================
    定义                          说明
    ===========================  ================================================
    `ModelType.MINDIR`           MindSpore模型的框架类型，该模型使用.mindir作为后缀。
    `ModelType.MINDIR_LITE`      MindSpore Lite模型的框架类型，该模型使用.ms作为后缀。
    ===========================  ================================================
