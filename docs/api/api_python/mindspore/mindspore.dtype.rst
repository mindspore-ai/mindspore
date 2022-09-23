mindspore.dtype
===============

.. py:class:: mindspore.dtype

    创建一个MindSpore数据类型的对象。

    `dtype` 的实际路径为 `/mindspore/common/dtype.py` 。运行以下命令导入环境：

    .. code-block::

        from mindspore import dtype as mstype

    - **数值型**

      目前，MindSpore支持 ``Int``，``Uint`` 和 ``Float`` 数据类型。详情请参照以下表格。

      ==============================================   =============================
      定义                                              描述
      ==============================================   =============================
      ``mindspore.int8`` ,  ``mindspore.byte``         8位整型数
      ``mindspore.int16`` ,  ``mindspore.short``       16位整型数
      ``mindspore.int32`` ,  ``mindspore.intc``        32位整型数
      ``mindspore.int64`` ,  ``mindspore.intp``        64位整型数
      ``mindspore.uint8`` ,  ``mindspore.ubyte``       无符号8位整型数
      ``mindspore.uint16`` ,  ``mindspore.ushort``     无符号16位整型数
      ``mindspore.uint32`` ,  ``mindspore.uintc``      无符号32位整型数
      ``mindspore.uint64`` ,  ``mindspore.uintp``      无符号64位整型数
      ``mindspore.float16`` ,  ``mindspore.half``      16位浮点数
      ``mindspore.float32`` ,  ``mindspore.single``    32位浮点数
      ``mindspore.float64`` ,  ``mindspore.double``    64位浮点数
      ``mindspore.complex64``                          64位复数
      ``mindspore.complex128``                         128位复数
      ==============================================   =============================

    - **其他类型**

      除数值型以外的其他数据类型，请参照以下表格。

      ============================   =================
      类型                            描述
      ============================   =================
      ``Tensor``                      MindSpore中的张量类型。数据格式采用NCHW。详情请参考 `tensor <https://www.gitee.com/mindspore/mindspore/blob/r1.9/mindspore/python/mindspore/common/tensor.py>`_ 。
      ``bool_``                       布尔型，值为 ``True`` 或者 ``False`` 。
      ``int_``                        整数标量。
      ``uint``                        无符号整数标量。
      ``float_``                      浮点标量。
      ``complex``                     复数标量。
      ``number``                      数值型，包括 ``int_``、``uint``、``float_``、``complex`` 和 ``bool_``。
      ``list_``                       由 ``tensor`` 构造的列表，例如 ``List[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
      ``tuple_``                      由 ``tensor`` 构造的元组，例如 ``Tuple[T0,T1,...,Tn]`` ，其中元素 ``Ti`` 可以是不同的类型。
      ``function``                    函数类型。两种返回方式，当function不是None时，直接返回function，另一种当function为None时返回function(参数: List[T0,T1,...,Tn]，返回值: T)。
      ``type_type``                   类型的类型定义。
      ``type_none``                   没有匹配的返回类型，对应 Python 中的 ``type(None)``。
      ``symbolic_key``                在 ``env_type`` 中用作变量的键的变量的值。
      ``env_type``                    用于存储函数的自由变量的梯度，其中键是自由变量节点的 `symbolic_key` ，值是梯度。
      ============================   =================

    - **树形拓扑**

      以上定义的数据类型遵从如下的树形拓扑结构：

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
              │   │    ├─── float16
              │   │    ├─── float32
              │   │    └─── float64
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
