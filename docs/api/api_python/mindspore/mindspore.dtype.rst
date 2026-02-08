mindspore.dtype
===============

.. py:class:: mindspore.dtype

    MindSpore的数据类型。

    `dtype` 的实际路径为 `/mindspore/common/dtype.py` 。运行以下命令导入环境：

    .. code-block::

        from mindspore import dtype as mstype

基础数据类型
^^^^^^^^^^^^^^^

MindSpore支持以下基础数据类型：

===================================================   =============================
定义                                                   描述
===================================================   =============================
``mindspore.bool``                                     布尔型
``mindspore.int8``                                     8位整型数
``mindspore.int16`` 、 ``mindspore.short``             16位整型数
``mindspore.int32`` 、 ``mindspore.int``               32位整型数
``mindspore.int64`` 、 ``mindspore.long``              64位整型数
``mindspore.uint8``                                    无符号8位整型数
``mindspore.uint16``                                   无符号16位整型数
``mindspore.uint32``                                   无符号32位整型数
``mindspore.uint64``                                   无符号64位整型数
``mindspore.float16`` 、 ``mindspore.half``            16位浮点数
``mindspore.float32`` 、 ``mindspore.float``           32位浮点数
``mindspore.float64`` 、 ``mindspore.double``          64位浮点数
``mindspore.bfloat16``                                 16位脑浮点数
``mindspore.complex64`` 、 ``mindspore.cfloat``        64位复数
``mindspore.complex128`` 、 ``mindspore.cdouble``      128位复数
===================================================   =============================

其他类型
^^^^^^^^^^^^^^^

除基础数据类型以外的其他数据类型，请参照以下表格。

============================   =================
类型                            描述
============================   =================
``Tensor``                      MindSpore中的张量类型。数据格式采用NCHW。详情请参考 `tensor <https://www.atomgit.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/tensor.py>`_ 。
``complex``                     复数标量。
``number``                      数值型，包括上述基础数据类型，例如 ``int16``、``float32``。
``list_``                       由 ``tensor`` 构造的列表，例如 ``List[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
``tuple_``                      由 ``tensor`` 构造的元组，例如 ``Tuple[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
``function``                    函数类型。两种返回方式：当function不是None时，直接返回function；当function为None时，返回function(参数：List[T0,T1,...,Tn]，返回值：T)。
``type_type``                   类型的类型定义。
``type_none``                   没有匹配的返回类型，对应 Python 中的 ``type(None)``。
``symbolic_key``                在 ``env_type`` 中用作变量的键的变量的值。
``env_type``                    用于存储函数的自由变量的梯度，其中键是自由变量节点的 `symbolic_key` ，值是梯度。
============================   =================

类型转换规则
^^^^^^^^^^^^^^^

当算子的部分输入要求目标类型一致时，会根据类型转换规则自动进行类型提升。如果这些输入存在不同大小和类别的类型（其中 ``complex > float > int > bool``），它们将会被提升为具有足够大小和类别的类型。

Tensor与Tensor的类型转换规则请参考以下表格。表格里首行和首列均表示输入 ``Tensor`` 的类型，表格中对应位置表示输出 ``Tensor`` 的类型， ``-`` 表示不进行类型提升。

为方便描述，表格中使用 ``bool`` 指代 ``mindspore.bool``，使用 ``int8`` 指代 ``mindspore.int8``，以此类推。

.. list-table::
    :widths: 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20
    :header-rows: 1

    * - Tensor与Tensor
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

Number与Tensor的类型转换规则请参考以下表格。表格里首行表示输入 ``Number`` 的类型，首列表示输入 ``Tensor`` 的类型。表格中对应位置表示输出 ``Tensor`` 的类型， ``-`` 表示不进行类型提升。

.. list-table::
    :widths: 20 20 20 20
    :header-rows: 1

    * - Number与Tensor
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
