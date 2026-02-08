mindspore.CSRTensor
===================

.. py:class:: mindspore.CSRTensor(indptr=None, indices=None, values=None, shape=None, csr_tensor=None)

    用来表示某一Tensor在给定索引上非零元素的集合，其中行索引由 `indptr` 表示，列索引由 `indices` 表示，非零值由 `values` 表示。

    如果 `indptr` 是[0, 2, 5, 6]， `indices` 是[0, 3, 1, 2, 4, 2]， `values` 是[1., 2., 3., 4., 5., 6.]， `shape` 是(3, 5)，那么它对应的稠密Tensor如下：

    .. code-block::

        [[1., 0., 0., 2., 0.],
         [0., 3., 4., 0., 5.],
         [0., 0., 6., 0., 0.]]

    `indptr` 的长度应为 `shape[0]+1`，其中的元素应为单调递增或相等的，最大值应等于张量中所有非零元素的总个数。 `indices` 和 `values` 的长度应等于张量中所有非零元素的总个数。具体地，通过 `indptr` 拿到每一行非零元素的查询索引，然后到 `indices` 中根据该行的查询索引确定该行非零元素所在列，到 `values` 中确定该行非零元素的取值。

    前例中， `indptr` 为[0, 2, 5, 6]，表示查询索引[0, 2)为第0行的数据，查询索引[2, 5)为第1行的数据，查询索引[5, 6)为第2行的数据。例如：

    - 稠密张量第0行非零元素所在的列位置由 `indices` 中的第[0, 2)个元素（即[0, 3]）给出，实际值由 `values` 中的第[0, 2)个元素（即[1., 2.]）给出。
    - 稠密张量第1行非零元素所在的列位置由 `indices` 中的第[2, 5)个元素（即[1, 2, 4]）给出，实际值由 `values` 中的第[2, 5)个元素（即[3., 4., 5.]）给出。
    - 稠密张量第2行非零元素所在的列位置由 `indices` 中的第[5, 6)个元素（即[2]）给出，实际值由 `values` 中的第[5, 6)个元素（即[6.]）给出。

    `CSRTensor` 的算术运算包括：加（+）、减（-）、乘（*）、除（/）。详细的算术运算支持请参考 `运算符 <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html#%E8%BF%90%E7%AE%97%E7%AC%A6>`_。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 如果 `indptr` 或 `indices` 的值不合法，行为将没有定义。不合法的值包括：当 `values` 或 `indices` 的长度超出了 `indptr` 所指定的取值范围、当 `indices` 在同一行中出现重复的列。

    参数：
        - **indptr** (Tensor, 可选) - shape为 :math:`(M)` 的一维整数Tensor，其中 :math:`M` 等于 `shape[0] + 1`，表示每行非零元素在 `values` 中存储的起止位置。支持的数据类型为int16、int32和int64。默认值： ``None`` 。
        - **indices** (Tensor, 可选) - shape为 :math:`(N)` 的一维整数Tensor，其中 :math:`N` 等于非零元素数量，表示每个元素的列索引值。支持的数据类型为int16、int32和int64。默认值： ``None`` 。
        - **values** (Tensor, 可选) - Tensor， `values` 的零维长度必须与 `indices` 的零维长度相等(values.shape[0] == indices.shape[0])。 `values` 用来表示索引对应的数值。默认值： ``None`` 。
        - **shape** (tuple(int), 可选) - shape为 :math:`(ndims)` 的整数元组，用来指定稀疏矩阵的稠密shape。`shape[0]` 表示行数，因此必须和 :math:`M - 1` 值相等。默认值： ``None`` 。
        - **csr_tensor** (CSRTensor, 可选) - CSRTensor对象，用来初始化新的CSRTensor。 `values` 的特征维度需要和 `csr_tensor` 的特征维度匹配 :math:`(values.shape[1:] == csr\_tensor.shape[2:])` 。默认值： ``None`` 。

    输出：
        CSRTensor，稠密shape取决于传入的 `shape` ，数据类型由 `values` 决定。

    .. py:method:: abs()

        对所有非零元素取绝对值，并返回新的CSRTensor。

        返回：
            CSRTensor。

    .. py:method:: add(b, alpha, beta)

        两个CSRTensor求和：C = alpha * a + beta * b。

        参数：
            - **b** (CSRTensor) - 稀疏CSRTensor。
            - **alpha** (Tensor) - 稠密Tensor，shape必须可以广播给self。
            - **beta** (Tensor) - 稠密Tensor，shape必须可以广播给 `b` 。

        返回：
            CSRTensor，求和。

    .. py:method:: astype(dtype)

        返回指定数据类型的CSRTensor。

        参数：
            - **dtype** (Union[mindspore.dtype, numpy.dtype, str]) - 指定数据类型。

        返回：
            CSRTensor。

    .. py:method:: dtype
        :property:

        返回稀疏矩阵非零元素值数据类型（:class:`mindspore.dtype`）。

    .. py:method:: indices
        :property:

        返回CSRTensor的列索引值。

    .. py:method:: indptr
        :property:

        返回CSRTensor的行偏移量。

    .. py:method:: itemsize
        :property:

        返回每个非零元素所占字节数。

    .. py:method:: mm(matrix)

        返回CSRTensor右乘稀疏矩阵或稠密矩阵的矩阵乘法运算结果。

        shape为 `[M, N]` 的CSRTensor，需要适配shape为 `[N, K]` 的稠密矩阵或稀疏矩阵，得到结果为 `[M, K]` 的稠密矩阵或稀疏矩阵。

        .. note::
            - 若右矩阵为Tensor，则仅支持安装了LLVM12.0.1及以上版本的CPU后端或GPU后端。
            - 若右矩阵为CSRTensor，则仅支持GPU后端。

        参数：
            - **matrix** (Tensor or CSRTensor) - shape为 `[N, K]` 的二维矩阵，其中N等于CSRTensor的列数。

        返回：
            Tensor 或者 CSRTensor。

    .. py:method:: mv(dense_vector)

        返回CSRTensor右乘稠密矩阵的矩阵乘法运算结果。
        shape为 `[M, N]` 的CSRTensor，需要适配shape为 `[N, 1]` 的稠密向量，得到结果为 `[M, 1]` 的稠密向量。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        参数：
            - **dense_vector** (Tensor) - shape为 `[N, 1]` 的二维Tensor，其中N等于CSRTensor的列数。

        返回：
            Tensor。

    .. py:method:: ndim
        :property:

        返回稀疏矩阵的稠密维度。

    .. py:method:: shape
        :property:

        返回稀疏矩阵的稠密shape。

    .. py:method:: size
        :property:

        返回稀疏矩阵非零元素值数量。

    .. py:method:: sum(axis)

        对CSRTensor的某个轴求和。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        参数：
            - **axis** (int) - 求和轴。

        返回：
            Tensor。

    .. py:method:: to_coo()

        将CSRTensor转换为COOTensor。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        返回：
            COOTensor。

    .. py:method:: to_dense()

        将CSRTensor转换为稠密Tensor。

        返回：
            Tensor。

    .. py:method:: to_tuple()

        将CSRTensor的行偏移量、列索引、非零元素，以及shape信息作为tuple返回。

        返回：
            tuple(Tensor, Tensor, Tensor, tuple(int))。

    .. py:method:: values
        :property:

        返回CSRTensor的非零元素值。
