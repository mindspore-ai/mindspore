mindspore.numpy.ravel_multi_index
=================================

.. py:function:: mindspore.numpy.ravel_multi_index(multi_index, dims, mode='clip', order='C')

    将元素为索引数组的tuple转换为展平的索引数组，并对多重索引应用边界模式。

    .. note::
        不支持 `raise` 模式。默认模式为 `clip` 。

    参数：
        - **multi_index** (类数组的tuple) - 一个元素为整数数组的tuple，每一维包含一个数组。
        - **dims** (Union[int, tuple(int)]) - `multi_index` 的索引将应用到的数组的shape。
        - **mode** ({`wrap`, `clip`}, 可选) - 指定如何处理越界索引。默认值： `'clip'` 。 `'wrap'` ：取越界索引除以轴长的余数。 `'clip'` ：裁剪到范围内。在 `'clip'` 模式下，取余后的负索引将裁剪至0。
        - **order** ({C, F}, 可选) - 确定多重索引是以行优先(C风格)还是列优先(Fortran风格)顺序查看。

    返回：
        展平的索引数组。表示 `dims` 维的展平的索引值。

    异常：
        - **TypeError** - 如果 `multi_index` 或 `dims` 不能转换为Tensor，或者 `dims` 不是整数值的序列。
        - **ValueError** - 如果 `multi_index` 和 `dims` 的长度不相等。

