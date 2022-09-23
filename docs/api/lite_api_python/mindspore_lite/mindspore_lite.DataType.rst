mindspore_lite.DataType
=======================

.. py:class:: mindspore_lite.DataType

    创建MindSpore Lite的数据类型对象。

    有关详细信息，请参见 `DataType <https://gitee.com/mindspore/mindspore/blob/r1.9/mindspore/lite/python/api/tensor.py>`_ 。
    运行以下命令导入包：

    .. code-block::

        from mindspore_lite import DataType

    * **类型**

      目前，MindSpore Lite支持"Int"类型、"Uint"类型和"Float"类型。
      下表列出了详细信息。

      ===========================  ================================================================
      定义                          说明
      ===========================  ================================================================
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
      ``DataType.INVALID``         ``DataType`` 的最大阈值，用于防止无效类型，对应于C++中的 ``INT32_MAX``
      ===========================  ================================================================

    * **用法**

      由于Python API中的 `mindspore_lite.Tensor` 是直接使用pybind11技术包装C++ API， `DataType` 在Python API和C++ API之间有一对一的对应关系，修改 `DataType` 的方法在 `tensor` 类的set和get方法中。

      - `set_data_type`: 在 `data_type_py_cxx_map` 中以Python API中的 `DataType` 为关键字进行查询，并获取C++ API中的 `DataType` ，将其传递给C++ API中的 `set_data_type` 方法。
      - `get_data_type`: 通过C++ API中的 `get_data_type` 方法在C++ API中获取 `DataType` ，以C++ API中的 `DataType` 为关键字在 `data_type_cxx_py_map` 中查询，返回在Python API中的 `DataType` 。

      以下是一个示例：

      .. code-block:: python

          from mindspore_lite import DataType
          from mindspore_lite import Tensor

          tensor = Tensor()
          tensor.set_data_type(DataType.FLOAT32)
          data_type = tensor.get_data_type()
          print(data_type)

      运行结果如下：

      .. code-block::

          DataType.FLOAT32
