mindspore.ops.SplitV
====================

.. py:class:: mindspore.ops.SplitV(size_splits, split_dim, num_split)

    沿给定维度将输入Tensor拆分为 `num_split` 个Tensor。

    `input_x` Tensor将被拆分为若干子Tensor，子Tensor的shape由 `size_splits` 沿拆分维度给出。
    这要求 `input_x.shape(split_dim)` 等于 `size_splits` 的总和。
    
    `input_x` 的shape为 :math:`(x_1, x_2, ..., x_M, ..., x_R)` 。 `input_x` 的秩为 `R` 。设
    给定的 `split_dim` 为 `M` ，同时 :math:`-R \le M < R` 。设给定的 `num_split` 为 `N` ，给定
    的 `size_splits` 为 :math:`(x_{m_1}, x_{m_2}, ..., x_{m_N})` ， :math:`x_M=\sum_{i=1}^Nx_{m_i}` 。
    输出为list(Tensor)，对于第 :math:`i` 个Tensor，其shape为 :math:`(x_1, x_2, ..., x_{m_i}, ..., x_R)` ，其中
    :math:`x_{m_i}` 是第 :math:`i` 个Tensor的第 :math:`M` 维。那么，输出Tensor的shape为：

    .. math::

        ((x_1, x_2, ..., x_{m_1}, ..., x_R), (x_1, x_2, ..., x_{m_2}, ..., x_R), ...,
         (x_1, x_2, ..., x_{m_N}, ..., x_R))

    参数：
        - **size_splits** (Union[tuple, list]) - 包含沿拆分维度的每个输出Tensor大小的list。
          必须与沿 `split_dim` 的值的维度和相等。可以包含一个-1，以表示要推断维度。
        - **split_dim** (int) - 沿着该维度进行拆分，必须在[-len(input_x.shape), len(input_x.shape))范围内。
        - **num_split** (int) - 输出Tensor的数量，必须是正整数。

    输入：
        - **input_x** (Tensor) - 该Tensor的shape为 :math:`(x_1, x_2, ...,x_M ..., x_R)` 。

    输出：
        Tensor，包含 `num_split` 个Tensor的list，其shape分别为
        :math:`((x_1, x_2, ..., x_{m_1}, ..., x_R),
        (x_1, x_2, ..., x_{m_2}, ..., x_R), ..., (x_1, x_2, ..., x_{m_N}, ..., x_R))` ，
        其中 :math:`x_M=\sum_{i=1}^Nx_{m_i}` ，数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `size_splits` 不是tuple或list。
        - **TypeError** - 如果 `size_splits` 的元素不是整数。
        - **TypeError** - 如果 `split_dim` 或 `num_split` 不是整数。
        - **ValueError** - 如果 `size_splits` 的秩不等于 `num_split` 。
        - **ValueError** - 如果 `size_splits` 的总和不等于值沿着 `split_dim` 的维度。
        - **ValueError** - 如果 `split_dim` 不在[-len(input_x.shape), len(input_x.shape))内。
        - **ValueError** - 如果 `num_split` 小于或等于0。
