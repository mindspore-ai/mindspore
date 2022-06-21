mindspore_lite.DataType
=======================

.. py:class:: mindspore_lite.DataType

  创建MindSpore Lite的数据类型对象。

  ``DataType`` 的实际路径是 ``/mindspore/lite/python/api/tensor.py``。
  运行以下命令导入包：

  .. code-block::

      from mindspore_lite import DataType

  * **类型**

    目前，MindSpore Lite支持“Int”类型、“Uint”类型和“Float”类型。
    下表列出了详细信息。

    ===========================  =============================
    定义                          说明
    ===========================  =============================
    ``DataType.UNKNOWN``         不匹配以下任何已知类型
    ``DataType.BOOL``            布尔值为 ``True`` 或 ``False``
    ``DataType.INT8``            8位整型数
    ``DataType.INT16``           16位整型数
    ``DataType.INT32``           32位整型数
    ``DataType.INT64``           64位整型数
    ``DataType.UINT8``           无符号8位整型数
    ``DataType.UINT16``          无符号16位整型数
    ``DataType.UINT32``          无符号32位整型数
    ``DataType.UINT64``          无符号64位整型数
    ``DataType.FLOAT16``         16位浮点数
    ``DataType.FLOAT32``         32位浮点数
    ``DataType.FLOAT64``         64位浮点数
    ``DataType.INVALID``         ``DataType``的最大阈值，用于防止无效类型，对应于C++中的 ``INT32_MAX``
    ===========================  =============================