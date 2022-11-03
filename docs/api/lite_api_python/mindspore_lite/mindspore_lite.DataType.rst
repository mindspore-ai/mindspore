mindspore_lite.DataType
=======================

.. py:class:: mindspore_lite.DataType

    `DataType` 类定义MindSpore Lite中Tensor的数据类型。

    目前，支持以下 `DataType` ：

    ===========================  =====================================
    定义                          说明
    ===========================  =====================================
    `DataType.UNKNOWN`           不匹配以下任何已知类型。
    `DataType.BOOL`              布尔值为 `True` 或 `False` 。
    `DataType.INT8`              8位整型数。
    `DataType.INT16`             16位整型数。
    `DataType.INT32`             32位整型数。
    `DataType.INT64`             64位整型数。
    `DataType.UINT8`             无符号8位整型数。
    `DataType.UINT16`            无符号16位整型数。
    `DataType.UINT32`            无符号32位整型数。
    `DataType.UINT64`            无符号64位整型数。
    `DataType.FLOAT16`           16位浮点数。
    `DataType.FLOAT32`           32位浮点数。
    `DataType.FLOAT64`           64位浮点数。
    `DataType.INVALID`           `DataType` 的最大阈值，用于防止无效类型。
    ===========================  =====================================
