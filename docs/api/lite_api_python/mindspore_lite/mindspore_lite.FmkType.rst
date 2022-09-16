mindspore_lite.FmkType
======================

.. py:class:: mindspore_lite.FmkType

    将第三方或MindSpore模型转换为MindSpore Lite模型时，FmkType定义输入模型的框架类型。

    有关详细信息，请参见 `FmkType <https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/python/api/converter.py>`_ 。
    运行以下命令导入包：

    .. code-block::

        from mindspore_lite import FmkType

    * **类型**

      目前，支持以下第三方模型框架类型：
      ``TF`` 类型, ``CAFFE`` 类型, ``ONNX`` 类型, ``MINDIR`` 类型和 ``TFLITE`` 类型。
      下表列出了详细信息。

      ===========================  ====================================================
      定义                          说明
      ===========================  ====================================================
      ``FmkType.TF``               TensorFlow模型的框架类型，该模型使用.pb作为后缀
      ``FmkType.CAFFE``            Caffe模型的框架类型，该模型使用.prototxt作为后缀
      ``FmkType.ONNX``             ONNX模型的框架类型，该模型使用.onnx作为后缀
      ``FmkType.MINDIR``           MindSpore模型的框架类型，该模型使用.mindir作为后缀
      ``FmkType.TFLITE``           TensorFlow Lite模型的框架类型，该模型使用.tflite作为后缀
      ``FmkType.PYTORCH``          PyTorch模型的框架类型，该模型使用.pt或.pth作为后缀
      ===========================  ====================================================
