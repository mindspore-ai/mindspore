mindspore_lite.ModelType
========================

.. py:class:: mindspore_lite.ModelType

    从文件加载或构建模型时，ModelType定义输入模型文件的类型。

    有关详细信息，请参见 `ModelType <https://gitee.com/mindspore/mindspore/blob/r1.9/mindspore/lite/python/api/model.py>`_ 。
    运行以下命令导入包：

    .. code-block::

        from mindspore_lite import ModelType

    * **类型**

      目前，支持以下第三方模型框架类型：
      ``ModelType.MINDIR`` 类型和 ``ModelType.MINDIR_LITE`` 类型。
      下表列出了详细信息。

      ===========================  ================================================
      定义                          说明
      ===========================  ================================================
      ``ModelType.MINDIR``         MindSpore模型的框架类型，该模型使用.mindir作为后缀
      ``ModelType.MINDIR_LITE``    MindSpore Lite模型的框架类型，该模型使用.ms作为后缀
      ===========================  ================================================
