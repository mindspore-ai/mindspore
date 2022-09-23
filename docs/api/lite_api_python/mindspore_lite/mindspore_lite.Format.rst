mindspore_lite.Format
=====================

.. py:class:: mindspore_lite.Format

    MindSpore Lite的“张量”类型。例如：格式。NCHW。

    有关详细信息，请参见 `Format <https://gitee.com/mindspore/mindspore/blob/r1.9/mindspore/lite/python/api/tensor.py>`_ 。
    运行以下命令导入包：

    .. code-block::

        from mindspore_lite import Format

    * **类型**

      有关支持的格式，请参见下表：

      ===========================  ===============================================
      定义                          说明
      ===========================  ===============================================
      ``Format.DEFAULT``           默认格式
      ``Format.NCHW``              按批次N、通道C、高度H和宽度W的顺序存储张量数据
      ``Format.NHWC``              按批次N、高度H、宽度W和通道C的顺序存储张量数据
      ``Format.NHWC4``             C轴4字节对齐格式的 ``Format.NHWC``
      ``Format.HWKC``              按高度H、宽度W、核数K和通道C的顺序存储张量数据
      ``Format.HWCK``              按高度H、宽度W、通道C和核数K的顺序存储张量数据
      ``Format.KCHW``              按核数K、通道C、高度H和宽度W的顺序存储张量数据
      ``Format.CKHW``              按通道C、核数K、高度H和宽度W的顺序存储张量数据
      ``Format.KHWC``              按核数K、高度H、宽度W和通道C的顺序存储张量数据
      ``Format.CHWK``              按通道C、高度H、宽度W和核数K的顺序存储张量数据
      ``Format.HW``                按高度H和宽度W的顺序存储张量数据
      ``Format.HW4``               w轴4字节对齐格式的 ``Format.HW``
      ``Format.NC``                按批次N和通道C的顺序存储张量数据
      ``Format.NC4``               C轴4字节对齐格式的 ``Format.NC``
      ``Format.NC4HW4``            C轴4字节对齐和W轴4字节对齐格式的 ``Format.NCHW``
      ``Format.NCDHW``             按批次N、通道C、深度D、高度H和宽度W的顺序存储张量数据
      ``Format.NWC``               按批次N、宽度W和通道C的顺序存储张量数据
      ``Format.NCW``               按批次N、通道C和宽度W的顺序存储张量数据
      ``Format.NDHWC``             按批次N、深度D、高度H、宽度W和通道C的顺序存储张量数据
      ``Format.NC8HW8``            C轴8字节对齐和W轴8字节对齐格式的 ``Format.NCHW``
      ===========================  ===============================================

    * **用法**

      由于Python API中的 `mindspore_lite.Tensor` 是直接使用pybind11技术包装C++ API， `Format` 在Python API和C++ API之间有一对一的对应关系，修改 `Format` 的方法在 `tensor` 类的set和get方法中。

      - `set_format`: 在 `format_py_cxx_map` 中以Python API中的 `Format` 为关键字进行查询，并获取C++ API中的 `Format` ，将其传递给C++ API中的 `set_format` 方法。
      - `get_format`: 通过C++ API中的 `get_format` 方法在C++ API中获取 `Format` ，以C++ API中的 `Format` 为关键字在 `format_cxx_py_map` 中查询，返回在Python API中的 `Format` 。

      以下是一个示例：

    .. code-block:: python

        from mindspore_lite import Format
        from mindspore_lite import Tensor

        tensor = Tensor()
        tensor.set_format(Format.NHWC)
        tensor_format = tensor.get_format()
        print(tensor_format)

      运行结果如下：

      .. code-block::

          Format.NHWC
