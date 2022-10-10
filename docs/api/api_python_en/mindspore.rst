mindspore
=========

Tensor
------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Tensor
    mindspore.COOTensor
    mindspore.CSRTensor
    mindspore.RowTensor
    mindspore.SparseTensor

Parameter
---------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Parameter
    mindspore.ParameterTuple

DataType
--------

.. class:: mindspore.dtype

  Create a data type object of MindSpore.

  The actual path of ``dtype`` is ``/mindspore/common/dtype.py``.
  Run the following command to import the package:

  .. code-block::

      from mindspore import dtype as mstype

  * **Numeric Type**

    Currently, MindSpore supports ``Int`` type, ``Uint`` type, ``Float`` type and ``Complex`` type.
    The following table lists the details.

    ==============================================   =============================
    Definition                                        Description
    ==============================================   =============================
    ``mindspore.int8`` ,  ``mindspore.byte``         8-bit integer
    ``mindspore.int16`` ,  ``mindspore.short``       16-bit integer
    ``mindspore.int32`` ,  ``mindspore.intc``        32-bit integer
    ``mindspore.int64`` ,  ``mindspore.intp``        64-bit integer
    ``mindspore.uint8`` ,  ``mindspore.ubyte``       unsigned 8-bit integer
    ``mindspore.uint16`` ,  ``mindspore.ushort``     unsigned 16-bit integer
    ``mindspore.uint32`` ,  ``mindspore.uintc``      unsigned 32-bit integer
    ``mindspore.uint64`` ,  ``mindspore.uintp``      unsigned 64-bit integer
    ``mindspore.float16`` ,  ``mindspore.half``      16-bit floating-point number
    ``mindspore.float32`` ,  ``mindspore.single``    32-bit floating-point number
    ``mindspore.float64`` ,  ``mindspore.double``    64-bit floating-point number
    ``mindspore.complex64``                          64-bit complex number
    ``mindspore.complex128``                         128-bit complex number
    ==============================================   =============================

  * **Other Type**

    For other defined types, see the following table.

    ============================   =================
    Type                            Description
    ============================   =================
    ``tensor``                      MindSpore's ``tensor`` type. Data format uses NCHW. For details, see `tensor <https://www.gitee.com/mindspore/mindspore/blob/r1.9/mindspore/python/mindspore/common/tensor.py>`_.
    ``bool_``                       Boolean ``True`` or ``False``.
    ``int_``                        Integer scalar.
    ``uint``                        Unsigned integer scalar.
    ``float_``                      Floating-point scalar.
    ``complex``                     Complex scalar.
    ``number``                      Number, including ``int_`` , ``uint`` , ``float_`` , ``complex`` and ``bool_`` .
    ``list_``                       List constructed by ``tensor`` , such as ``List[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
    ``tuple_``                      Tuple constructed by ``tensor`` , such as ``Tuple[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
    ``function``                    Function. Return in two ways, when function is not None, returns Func directly, the other returns Func(args: List[T0,T1,...,Tn], retval: T) when function is None.
    ``type_type``                   Type definition of type.
    ``type_none``                   No matching return type, corresponding to the ``type(None)`` in Python.
    ``symbolic_key``                The value of a variable is used as a key of the variable in ``env_type`` .
    ``env_type``                    Used to store the gradient of the free variable of a function, where the key is the ``symbolic_key`` of the free variable's node and the value is the gradient.
    ============================   =================

  * **Tree Topology**

    The relationships of the above types are as follows:

    .. code-block::


        └─────── number
            │   ├─── bool_
            │   ├─── int_
            │   │   ├─── int8, byte
            │   │   ├─── int16, short
            │   │   ├─── int32, intc
            │   │   └─── int64, intp
            │   ├─── uint
            │   │   ├─── uint8, ubyte
            │   │   ├─── uint16, ushort
            │   │   ├─── uint32, uintc
            │   │   └─── uint64, uintp
            │   ├─── float_
            │   │   ├─── float16
            │   │   ├─── float32
            │   │   └─── float64
            │   └─── complex
            │       ├─── complex64
            │       └─── complex128
            ├─── tensor
            │   ├─── Array[Float32]
            │   └─── ...
            ├─── list_
            │   ├─── List[Int32,Float32]
            │   └─── ...
            ├─── tuple_
            │   ├─── Tuple[Int32,Float32]
            │   └─── ...
            ├─── function
            │   ├─── Func
            │   ├─── Func[(Int32, Float32), Int32]
            │   └─── ...
            ├─── type_type
            ├─── type_none
            ├─── symbolic_key
            └─── env_type


.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dtype_to_nptype
    mindspore.issubclass_
    mindspore.dtype_to_pytype
    mindspore.pytype_to_dtype
    mindspore.get_py_obj_dtype

Seed
----

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.set_seed
    mindspore.get_seed

Context
--------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.set_context
    mindspore.get_context
    mindspore.set_auto_parallel_context
    mindspore.get_auto_parallel_context
    mindspore.reset_auto_parallel_context
    mindspore.ParallelMode
    mindspore.set_ps_context
    mindspore.get_ps_context
    mindspore.reset_ps_context
    mindspore.set_algo_parameters
    mindspore.get_algo_parameters
    mindspore.reset_algo_parameters

Model
-----

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Model

Callback
--------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Callback
    mindspore.CheckpointConfig
    mindspore.EarlyStopping
    mindspore.History
    mindspore.LambdaCallback
    mindspore.LearningRateScheduler
    mindspore.LossMonitor
    mindspore.ModelCheckpoint
    mindspore.ReduceLROnPlateau
    mindspore.RunContext
    mindspore.TimeMonitor

Dataset Helper
---------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.DatasetHelper
    mindspore.connect_network_with_dataset
    mindspore.data_sink

Serialization
-------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.async_ckpt_thread_status
    mindspore.build_searched_strategy
    mindspore.convert_model
    mindspore.export
    mindspore.load
    mindspore.load_checkpoint
    mindspore.load_distributed_checkpoint
    mindspore.load_param_into_net
    mindspore.merge_sliced_parameter
    mindspore.parse_print
    mindspore.restore_group_info_list
    mindspore.save_checkpoint

JIT
---

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.JitConfig
    mindspore.ms_function
    mindspore.ms_class
    mindspore.mutable

Log
---

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.get_level
    mindspore.get_log_config

Installation Verification
--------------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.run_check

Debugging and Tuning
--------------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.Profiler
    mindspore.SummaryCollector
    mindspore.SummaryLandscape
    mindspore.SummaryRecord
    mindspore.set_dump

Memory Recycle
--------------------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ms_memory_recycle

Thor
---------------

.. autosummary::
    :toctree: mindspore
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ConvertModelUtils
    mindspore.ConvertNetUtils
