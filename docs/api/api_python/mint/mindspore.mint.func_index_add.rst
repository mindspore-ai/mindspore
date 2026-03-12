mindspore.mint.index_add
========================

.. py:function:: mindspore.mint.index_add(input, dim, index, source, *, alpha=1) -> Tensor

    根据 `index` 中的索引顺序，将 `alpha` 乘以 `source` 的元素累加到 `input` 中。例如，如果 ``dim == 0``，``index[i] == j``，且 ``alpha = -1``，那么 `source` 的第 ``i`` 行将从 `input` 的第 ``j`` 行中被减去。`source` 的第 `dim` 维度必须与 `index` 的长度相同，且所有其他维度必须与 `input` 匹配，否则将引发错误。对于一个三维张量，输出定义如下：

    .. math::
        \begin{array}{ll}
        input[index[i],\ :,\ :]\ +=\ alpha * source[i,\ :,\ :]  \qquad \#if\ dim == 0 \\
        input[:,\ \ index[i],\ :]\ +=\ alpha * source[:,\ \ i,\ :]  \qquad \#if\ dim == 1 \\
        input[:,\ :,\ \ index[i]]\ +=\ alpha * source[:,\ :,\ \ i]  \qquad\#if\ dim == 2 \\
        \end{array}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定 `index` 属于哪个维度。
        - **index** (Tensor) - 指定 `input` 和 `source` 在轴 `dim` 的指定下标位置相加，数据类型为 `int32`。要求 `index` shape的维度为一维，并且 `index` shape的大小与 `source` shape在 `dim` 轴上的大小一致。 `index` 中元素取值范围为[0, b)，其中b的值为 `input` shape在 `dim` 轴上的大小。
        - **source** (Tensor) - 要进行相加的输入tensor，一定要与 `input` 有相同的数据类型，与 `input` 在 `dim` 维度有相同的shape。

    关键字参数：
        - **alpha** (number，可选) - `source` 的乘数。默认值： ``1`` 。

    返回：
        相加后的Tensor。shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `index` 或者 `source` 的类型不是Tensor。
        - **ValueError** - `dim` 的值超出 `source` shape的维度范围。
        - **ValueError** - `index` shape的维度和 `source` shape的维度不一致。
        - **ValueError** - `index` shape的维度不是1D，或者 `index` shape的大小与 `source` shape在 `dim` 轴上的大小不一致。
        - **ValueError** - 除 `dim` 轴外， `input` shape和 `source` shape的大小不一致。
