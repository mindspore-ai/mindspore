mindspore_lite
==============

Context
--------

.. autosummary::
    :toctree: mindspore_lite
    :nosignatures:
    :template: classtemplate.rst

    mindspore_lite.Context
    mindspore_lite.DeviceInfo
    mindspore_lite.CPUDeviceInfo
    mindspore_lite.GPUDeviceInfo
    mindspore_lite.AscendDeviceInfo

Converter
--------

.. py:class:: mindspore_lite.FmkType

  When converting a third-party or MindSpore model to a MindSpore Lite model, FmkType defines Input model's framework type.

  The actual path of ``FmkType`` is ``/mindspore/lite/python/api/converter.py``.
  Run the following command to import the package:

  .. code-block::

      from mindspore_lite import FmkType

  * **Type**

    Currently, the following third-party model framework types are supported:
    ``TF`` type, ``CAFFE`` type, ``ONNX`` type, ``MINDIR`` type, ``TFLITE`` type and ``PYTORCH`` type.
    The following table lists the details.

    ===========================  =============================
    Definition                    Description
    ===========================  =============================
    ``FmkType.TF``               TensorFlow model's framework type, and the model uses .pb as suffix
    ``FmkType.CAFFE``            Caffe model's framework type, and the model uses .prototxt as suffix
    ``FmkType.ONNX``             ONNX model's framework type, and the model uses .onnx as suffix
    ``FmkType.MINDIR``           MindSpore model's framework type, and the model uses .mindir as suffix
    ``FmkType.TFLITE``           TensorFlow Lite model's framework type, and the model uses .tflite as suffix
    ``FmkType.PYTORCH``          PYTORCH model's framework type, and the model uses .pt or .pth as suffix
    ===========================  =============================

.. autosummary::
    :toctree: mindspore_lite
    :nosignatures:
    :template: classtemplate.rst

    mindspore_lite.Converter

Model
-----

.. py:class:: mindspore_lite.ModelType

  When loading or building a model from file, ModelType defines the type of input model file.

  The actual path of ``ModelType`` is ``/mindspore/lite/python/api/model.py``.
  Run the following command to import the package:

  .. code-block::

      from mindspore_lite import ModelType

  * **Type**

    Currently, the following type of input model file are supported:
    ``ModelType.MINDIR`` type and ``ModelType.MINDIR_LITE`` type.
    The following table lists the details.

    ===========================  =============================
    Definition                    Description
    ===========================  =============================
    ``ModelType.MINDIR``         MindSpore model's type, which model uses .mindir as suffix
    ``ModelType.MINDIR_LITE``    MindSpore Lite model's type, which model uses .ms as suffix
    ===========================  =============================

.. autosummary::
    :toctree: mindspore_lite
    :nosignatures:
    :template: classtemplate.rst

    mindspore_lite.Model
    mindspore_lite.RunnerConfig
    mindspore_lite.ModelParallelRunner

Tensor
------

.. py:class:: mindspore_lite.DataType

  Create a data type object of MindSporeLite.

  The actual path of ``DataType`` is ``/mindspore/lite/python/api/tensor.py``.
  Run the following command to import the package:

  .. code-block::

      from mindspore_lite import DataType

  * **Type**

    Currently, MindSpore Lite supports ``Int`` type, ``Uint`` type and ``Float`` type.
    The following table lists the details.

    ===========================  =============================
    Definition                    Description
    ===========================  =============================
    ``DataType.UNKNOWN``         No matching any of the following known types.
    ``DataType.BOOL``            Boolean ``True`` or ``False``
    ``DataType.INT8``            8-bit integer
    ``DataType.INT16``           16-bit integer
    ``DataType.INT32``           32-bit integer
    ``DataType.INT64``           64-bit integer
    ``DataType.UINT8``           unsigned 8-bit integer
    ``DataType.UINT16``          unsigned 16-bit integer
    ``DataType.UINT32``          unsigned 32-bit integer
    ``DataType.UINT64``          unsigned 64-bit integer
    ``DataType.FLOAT16``         16-bit floating-point number
    ``DataType.FLOAT32``         32-bit floating-point number
    ``DataType.FLOAT64``         64-bit floating-point number
    ``DataType.INVALID``         The maximum threshold value of DataType to prevent invalid types, corresponding to the INT32_MAX in C++.
    ===========================  =============================

  * **Usage**

    Since `mindspore_lite.Tensor` in python api directly wraps c++ api with pybind11 technology, `DataType` has a one-to-one correspondence between the python api and the c++ api, and the way to modify `DataType` is in the set and to get methods of the `tensor` class. These include:

    - `set_data_type`: Query in `data_type_py_cxx_map` with `DataType` in python api as key, and get `DataType` in c++ api, pass it to `set_data_type` method in c++ api.
    - `get_data_type`: Get `DataType` in c++ api by `get_data_type` method in c++ api, Query in `data_type_cxx_py_map` with `DataType` in c++ api as key, return `DataType` in python api.

    Here is an example:

    .. code-block:: python

        from mindspore_lite import DataType
        from mindspore_lite import Tensor

        tensor = Tensor()
        tensor.set_data_type(DataType.FLOAT32)
        data_type = tensor.get_data_type()
        print(data_type)

    The result is as follows:

    .. code-block::

        DataType.FLOAT32

.. py:class:: mindspore_lite.Format

  MindSpore Lite's ``tensor`` type. For example: Format.NCHW.

  For details, see `tensor <https://www.gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/tensor.py>`_.
  Run the following command to import the package:

  .. code-block::

      from mindspore_lite import Format

  * **Type**

    See the following table for supported formats:

    ===========================  =============================
    Definition                    Description
    ===========================  =============================
    ``Format.DEFAULT``           default format
    ``Format.NCHW``              Store tensor data in the order of batch N, channel C, height H and width W
    ``Format.NHWC``              Store tensor data in the order of batch N, height H, width W and channel C
    ``Format.NHWC4``             C-axis 4-byte aligned Format.NHWC
    ``Format.HWKC``              Store tensor data in the order of height H, width W, kernel num K and channel C
    ``Format.HWCK``              Store tensor data in the order of height H, width W, channel C and kernel num K
    ``Format.KCHW``              Store tensor data in the order of kernel num K, channel C, height H and width W
    ``Format.CKHW``              Store tensor data in the order of channel C, kernel num K, height H and width W
    ``Format.KHWC``              Store tensor data in the order of kernel num K, height H, width W and channel C
    ``Format.CHWK``              Store tensor data in the order of channel C, height H, width W and kernel num K
    ``Format.HW``                Store tensor data in the order of height H and width W
    ``Format.HW4``               w-axis 4-byte aligned Format.HW
    ``Format.NC``                Store tensor data in the order of batch N and channel C
    ``Format.NC4``               C-axis 4-byte aligned Format.NC
    ``Format.NC4HW4``            C-axis 4-byte aligned and W-axis 4-byte aligned Format.NCHW
    ``Format.NCDHW``             Store tensor data in the order of batch N, channel C, depth D, height H and width W
    ``Format.NWC``               Store tensor data in the order of batch N, width W and channel C
    ``Format.NCW``               Store tensor data in the order of batch N, channel C and width W
    ``Format.NDHWC``             Store tensor data in the order of batch N, depth D, height H, width W and channel C
    ``Format.NC8HW8``            C-axis 8-byte aligned and W-axis 8-byte aligned Format.NCHW
    ===========================  =============================

  * **Usage**

    Since `mindspore_lite.Tensor` in python api directly wraps c++ api with pybind11 technology, `Format` has a one-to-one correspondence between the python api and the c++ api, and the way to modify `Format` is in the set and get methods of the `tensor` class. These includes:

    - `set_format`: Query in `format_py_cxx_map` with `Format` in python api as key, and get `Format` in c++ api, pass it to `set_format` method in c++ api.
    - `get_format`: Get `Format` in c++ api by `get_format` method in c++ api, Query in `format_cxx_py_map` with `Format` in c++ api as key, return `Format` in python api.

    Here is an example:

    .. code-block:: python

        from mindspore_lite import Format
        from mindspore_lite import Tensor

        tensor = Tensor()
        tensor.set_format(Format.NHWC)
        tensor_format = tensor.get_format()
        print(tensor_format)

    The result is as follows:

    .. code-block::

        Format.NHWC

.. autosummary::
    :toctree: mindspore_lite
    :nosignatures:
    :template: classtemplate.rst

    mindspore_lite.Tensor


