mindspore.COOTensor
===================

.. py:class:: mindspore.COOTensor(indices=None, values=None, shape=None, coo_tensor=None)

    用来表示某一Tensor在给定索引上非零元素的集合，其中索引(indices)指示了每一个非零元素的位置。

    对一个稠密Tensor `dense` 来说，它对应的COOTensor(indices, values, shape)，满足 `dense[indices[i]] = values[i]` 。

    如果 `indices` 是[[0, 1], [1, 2]]， `values` 是[1, 2]， `shape` 是(3, 4)，那么它对应的稠密Tensor如下：

    .. code-block::

        [[0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 0]]


    COOTensor的算术运算包括：加（+）、减（-）、乘（*）、除（/）。详细的算术运算支持请参考 `运算符 <https://www.mindspore.cn/docs/zh-CN/master/note/static_graph_syntax_support.html#%E8%BF%90%E7%AE%97%E7%AC%A6>`_。

    .. note::
        这是一个实验特性，在未来可能会发生API的变化。目前COOTensor中相同索引的值不会进行合并。如果索引中包含界外值，则得出未定义结果。

    参数：
        - **indices** (Tensor) - shape为 :math:`(N, ndims)` 的二维整数Tensor，其中N和ndims分别表示稀疏Tensor中 `values` 的数量和COOTensor维度的数量。目前 `ndims` 只能为2。请确保indices的值在所给shape范围内。支持的数据类型为int16， int32和int64。
        - **values** (Tensor) - shape为 :math:`(N)` 的一维Tensor，用来给 `indices` 中的每个元素提供数值。
        - **shape** (tuple(int)) - shape为ndims的整数元组，用来指定稀疏矩阵的稠密shape。
        - **coo_tensor** (COOTensor) - COOTensor对象，用来初始化新的COOTensor。

    返回：
        COOTensor，由 `indices` 、 `values` 和 `shape` 组成。

    .. py:method:: abs()

        对所有非零元素取绝对值，并返回新的COOTensor。

        返回：
            COOTensor。

    .. py:method:: add(other: COOTensor, thresh: Tensor)

        与另一个COOTensor相加，并返回新的COOTensor。

        参数：
            - **other** (COOTensor) - 另一个操作数，与当前操作数相加。
            - **thresh** (Tensor) - 0维，用来决定COOTensor.add结果中的indices/values对是否出现。

        返回：
            COOTensor，为两COOTensor相加后的结果。

        异常：
            - **ValueError** - 如果操作数(本COOTensor/other)的indices的维度不等于2。
            - **ValueError** - 如果操作数(本COOTensor/other)的values的维度不等于1。
            - **ValueError** - 如果操作数(本COOTensor/other)的shape的维度不等于1。
            - **ValueError** - 如果thresh的维度不等于0。
            - **TypeError** - 如果操作数(本COOTensor/other)的indices的数据类型不为int64。
            - **TypeError** - 如果操作数(本COOTensor/other)的shape的数据类型不为int64。
            - **ValueError** - 如果操作数(本COOTensor/other)的indices的长度不等于它的values的长度。
            - **TypeError** - 如果操作数(本COOTensor/other)的values的数据类型不为(int8/int16/int32/int64/float32/float64/complex64/complex128)中的任何一个。
            - **TypeError** - 如果thresh的数据类型不为(int8/int16/int32/int64/float32/float64)中的任何一个。
            - **TypeError** - 如果操作数(本COOTensor)的indices数据类型不等于other的indices数据类型。
            - **TypeError** - 如果操作数(本COOTensor)的values数据类型不等于other的values数据类型。
            - **TypeError** - 如果操作数(本COOTensor)的shape数据类型不等于other的shape数据类型。
            - **TypeError** - 如果操作数(本COOTensor/other)的values的数据类型与thresh数据类型不匹配。

    .. py:method:: astype(dtype: mstype)

        返回指定数据类型的COOTensor。

        参数：
            - **dtype** (Union[mindspore.dtype, numpy.dtype, str]) - 指定数据类型。

        返回：
            COOTensor。

    .. py:method:: coalesce()

        合并COOTensor中相同索引的值。

        返回：
            COOTensor。

    .. py:method:: dtype
        :property:

        返回COOTensor数据类型（:class:`mindspore.dtype`）。

    .. py:method:: indices
        :property:

        返回COOTensor的索引值。

    .. py:method:: itemsize
        :property:

        返回每个非零元素所占字节数。

    .. py:method:: ndim
        :property:

        返回稀疏矩阵的稠密维度。

    .. py:method:: shape
        :property:

        返回稀疏矩阵的稠密shape。

    .. py:method:: size
        :property:

        返回稀疏矩阵非零元素值数量。

    .. py:method:: to_csr()

        将COOTensor转换为CSRTensor。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        返回：
            CSRTensor。

    .. py:method:: to_dense()

        将COOTensor转换为稠密Tensor。

        返回：
            Tensor。

    .. py:method:: to_tuple()

        将COOTensor的索引，非零元素，以及shape信息作为tuple返回。

        返回：
            tuple(Tensor, Tensor, tuple(int))。

    .. py:method:: values
        :property:

        返回COOTensor的非零元素值。

