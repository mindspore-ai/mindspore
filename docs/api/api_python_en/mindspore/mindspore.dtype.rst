mindspore.dtype
===============

.. class:: mindspore.dtype

    Data type for MindSpore.

    The actual path of ``dtype`` is ``/mindspore/common/dtype.py``.
    Run the following command to import the package:

    .. code-block::

        from mindspore import dtype as mstype

Basic Data Type
^^^^^^^^^^^^^^^

MindSpore supports the following base data types:

===================================================   =============================
Definition                                             Description
===================================================   =============================
``mindspore.bool``                                     Boolean
``mindspore.int8``                                     8-bit integer
``mindspore.int16`` ,  ``mindspore.short``             16-bit integer
``mindspore.int32`` ,  ``mindspore.int``               32-bit integer
``mindspore.int64`` ,  ``mindspore.long``              64-bit integer
``mindspore.uint8``                                    unsigned 8-bit integer
``mindspore.uint16``                                   unsigned 16-bit integer
``mindspore.uint32``                                   unsigned 32-bit integer
``mindspore.uint64``                                   unsigned 64-bit integer
``mindspore.float16`` ,  ``mindspore.half``            16-bit floating-point number
``mindspore.float32`` ,  ``mindspore.float``           32-bit floating-point number
``mindspore.float64`` ,  ``mindspore.double``          64-bit floating-point number
``mindspore.bfloat16``                                 16-bit brain-floating-point number
``mindspore.complex64`` ,  ``mindspore.cfloat``        64-bit complex number
``mindspore.complex128`` ,  ``mindspore.cdouble``      128-bit complex number
===================================================   =============================

Other Type
^^^^^^^^^^^^^^^

For other defined types, see the following table.

============================   =================
Type                            Description
============================   =================
``tensor``                      MindSpore's ``tensor`` type. Data format uses NCHW. For details, see `tensor <https://www.gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/tensor.py>`_.
``complex``                     Complex scalar.
``number``                      Number, including base data types above, such as ``int16`` , ``float32`` .
``list_``                       List constructed by ``tensor`` , such as ``List[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
``tuple_``                      Tuple constructed by ``tensor`` , such as ``Tuple[T0,T1,...,Tn]`` , where the element ``Ti`` can be of different types.
``function``                    Function. Returns in two ways: when function is not None, it returns Func directly; otherwise, it returns Func(args: List[T0,T1,...,Tn], retval: T) when function is None.
``type_type``                   Type definition of type.
``type_none``                   No matching return type, corresponding to the ``type(None)`` in Python.
``symbolic_key``                The value of a variable is used as a key of the variable in ``env_type`` .
``env_type``                    Used to store the gradient of the free variable of a function, where the key is the ``symbolic_key`` of the free variable's node and the value is the gradient.
============================   =================

Type conversion rules
^^^^^^^^^^^^^^^^^^^^^^^

When some inputs of an operator are required to have the same target type, type promotion will be automatically performed according to the type conversion rules. If these inputs have types of different sizes and categories (where ``complex > float > int > bool`` ), they will be promoted to a type with sufficient size and category.

For the type conversion rules between Tensor and Tensor, please refer to the following table. The first row and the first column in the table both represent the types of the input ``Tensor``, and the corresponding position in the table represents the type of the output ``Tensor``. ``-`` indicates that no type promotion will be performed.

For convenience of description, ``bool`` is used in the table to refer to ``mindspore.bool``, ``int8`` is used to refer to ``mindspore.int8``, and so on.

.. list-table::
    :widths: 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20
    :header-rows: 1

    * - Tensor and Tensor
      - **bool**
      - **int8**
      - **int16**
      - **int32**
      - **int64**
      - **uint8**
      - **uint16**
      - **uint32**
      - **uint64**
      - **float16**
      - **bfloat16**
      - **float32**
      - **float64**
      - **complex64**
      - **complex128**
    * - **bool**
      - ``bool``
      - ``int8``
      - ``int16``
      - ``int32``
      - ``int64``
      - ``uint8``
      - ``uint16``
      - ``uint32``
      - ``uint64``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **int8**
      - ``int8``
      - ``int8``
      - ``int16``
      - ``int32``
      - ``int64``
      - ``int16``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **int16**
      - ``int16``
      - ``int16``
      - ``int16``
      - ``int32``
      - ``int64``
      - ``int16``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **int32**
      - ``int32``
      - ``int32``
      - ``int32``
      - ``int32``
      - ``int64``
      - ``int32``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **int64**
      - ``int64``
      - ``int64``
      - ``int64``
      - ``int64``
      - ``int64``
      - ``int64``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **uint8**
      - ``uint8``
      - ``int16``
      - ``int16``
      - ``int32``
      - ``int64``
      - ``uint8``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **uint16**
      - ``uint16``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``uint16``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
    * - **uint32**
      - ``uint32``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``uint32``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
    * - **uint64**
      - ``uint64``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``uint64``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
      - ``-``
    * - **float16**
      - ``float16``
      - ``float16``
      - ``float16``
      - ``float16``
      - ``float16``
      - ``float16``
      - ``-``
      - ``-``
      - ``-``
      - ``float16``
      - ``float32``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **bfloat16**
      - ``bfloat16``
      - ``bfloat16``
      - ``bfloat16``
      - ``bfloat16``
      - ``bfloat16``
      - ``bfloat16``
      - ``-``
      - ``-``
      - ``-``
      - ``float32``
      - ``bfloat16``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **float32**
      - ``float32``
      - ``float32``
      - ``float32``
      - ``float32``
      - ``float32``
      - ``float32``
      - ``-``
      - ``-``
      - ``-``
      - ``float32``
      - ``float32``
      - ``float32``
      - ``float64``
      - ``complex64``
      - ``complex128``
    * - **float64**
      - ``float64``
      - ``float64``
      - ``float64``
      - ``float64``
      - ``float64``
      - ``float64``
      - ``-``
      - ``-``
      - ``-``
      - ``float64``
      - ``float64``
      - ``float64``
      - ``float64``
      - ``complex128``
      - ``complex128``
    * - **complex64**
      - ``complex64``
      - ``complex64``
      - ``complex64``
      - ``complex64``
      - ``complex64``
      - ``complex64``
      - ``-``
      - ``-``
      - ``-``
      - ``complex64``
      - ``complex64``
      - ``complex64``
      - ``complex128``
      - ``complex64``
      - ``complex128``
    * - **complex128**
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``-``
      - ``-``
      - ``-``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``
      - ``complex128``


For the type conversion rules between Number and Tensor, please refer to the following table. The first row in the table indicates the type of the input ``Number``, and the first column indicates the types of input ``Tensor``. The corresponding position in the table represents the type of the output ``Tensor``. ``-`` indicates that no type promotion will be performed.

.. list-table::
    :widths: 20 20 20 20
    :header-rows: 1

    * - Number and Tensor
      - **bool**
      - **int**
      - **float**
    * - **bool**
      - ``bool``
      - ``int64``
      - ``float32``
    * - **int8**
      - ``int8``
      - ``int8``
      - ``float32``
    * - **int16**
      - ``int16``
      - ``int16``
      - ``float32``
    * - **int32**
      - ``int32``
      - ``int32``
      - ``float32``
    * - **int64**
      - ``int64``
      - ``int64``
      - ``float32``
    * - **uint8**
      - ``uint8``
      - ``uint8``
      - ``float32``
    * - **uint16**
      - ``uint16``
      - ``-``
      - ``-``
    * - **uint32**
      - ``uint32``
      - ``-``
      - ``-``
    * - **uint64**
      - ``uint64``
      - ``-``
      - ``-``
    * - **float16**
      - ``float16``
      - ``float16``
      - ``float16``
    * - **bfloat16**
      - ``bfloat16``
      - ``bfloat16``
      - ``bfloat16``
    * - **float32**
      - ``float32``
      - ``float32``
      - ``float32``
    * - **float64**
      - ``float64``
      - ``float64``
      - ``float64``
    * - **complex64**
      - ``complex64``
      - ``complex64``
      - ``complex64``
    * - **complex128**
      - ``complex128``
      - ``complex128``
      - ``complex128``
