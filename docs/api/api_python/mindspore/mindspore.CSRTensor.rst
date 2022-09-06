mindspore.CSRTensor
===================

.. py:class:: mindspore.CSRTensor(indptr=None, indices=None, values=None, shape=None, csr_tensor=None)

    用来表示某一张量在给定索引上非零元素的集合，其中行索引由 `indptr` 表示，列索引由 `indices` 
    表示，非零值由 `values` 表示。

    如果 `indptr` 是[0, 1, 2, 2]， `indices` 是[1, 2]， `values` 是[1., 2.]， `shape` 是(3, 4)，那么它对应的稠密Tensor如下：

    .. code-block::

        [[0., 1., 0., 0.],
         [0., 0., 2., 0.],
         [0., 0., 0., 0.]]

    .. note::
        这是一个实验特性，在未来可能会发生API的变化。
        如果values或indices的长度超出了indptr所指定的取值范围，超出部分的行为将没有定义。

    参数：
        - **indptr** (Tensor) - 形状为 `[M]` 的一维整数张量，其中M等于 `shape[0] + 1` , 表示每行非零元素的在 `values` 中存储的起止位置。默认值：None。支持的数据类型为 `int16` ， `int32` 和 `int64` 。
        - **indices** (Tensor) - 形状为 `[N]` 的一维整数张量，其中N等于非零元素数量，表示每个元素的列索引值。默认值：None。支持的数据类型为 `int16` ， `int32` 和 `int64` 。
        - **values** (Tensor) - 张量，values的0维长度必须与indices的0维长度相等(values.shape[0] == indices.shape[0])。values用来表示索引对应的数值。默认值：None。
        - **shape** (tuple(int)) - 形状为ndims的整数元组，用来指定稀疏矩阵的稠密形状。`shape[0]` 表示行数，因此必须和 `M - 1` 值相等。默认值：None。
        - **csr_tensor** (CSRTensor) - CSRTensor对象，用来初始化新的CSRTensor。values的特征维度需要和csr_tensor的特征维度匹配(values.shape[1:] == csr_tensor.shape[2:])。默认值：None。

    输出：
        CSRTensor，稠密形状取决于传入的 `shape` ，数据类型由 `values` 决定。

    .. py:method:: abs()

        对所有非零元素取绝对值，并返回新的CSRTensor。

        返回：
            CSRTensor。
        
    .. py:method:: add(b, alpha, beta)

        两个CSR张量求和：C = alpha * a + beta * b

        参数：
            - **b** (CSRTensor) - 稀疏CSR张量。
            - **alpha** (Tensor) - 稠密张量，shape必须可以广播给self。
            - **beta** (Tensor) - 稠密张量，shape必须可以广播给 `b` 。

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

    .. py:method:: mm(dense_matrix)

        返回CSRTensor右乘稠密矩阵的矩阵乘法运算结果。
        形状为 `[M, N]` 的CSRTensor，需要适配形状为 `[N, K]` 的稠密矩阵，得到结果为 `[M, K]` 的稠密矩阵。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        参数：
            - **dense_matrix** (Tensor) - 形状为 `[N，K]` 的二维矩阵，其中N等于CSRTensor的列数。

        返回：
            Tensor。

    .. py:method:: mv(dense_vector)

        返回CSRTensor右乘稠密矩阵的矩阵乘法运算结果。
        形状为 `[M, N]` 的CSRTensor，需要适配形状为 `[N, 1]` 的稠密向量，得到结果为 `[M, 1]` 的稠密向量。

        .. note::
            如果运行后端是CPU，那么仅支持在安装了LLVM12.0.1的机器运行。

        参数：
            - **dense_vector** (Tensor) - 形状为 `[N，1]` 的二维张量，其中N等于CSRTensor的列数。

        返回：
            Tensor。

    .. py:method:: ndim
        :property:

        稀疏矩阵的稠密维度。

    .. py:method:: shape
        :property:

        返回稀疏矩阵的稠密形状。

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

        将CSRTensor的行偏移量，列索引，非零元素，以及形状信息作为tuple返回。

        返回：
            tuple(Tensor，Tensor, Tensor, tuple(int))。

    .. py:method:: values
        :property:

        返回CSRTensor的非零元素值。
