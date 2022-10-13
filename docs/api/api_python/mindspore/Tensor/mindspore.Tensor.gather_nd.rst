mindspore.Tensor.gather_nd
==========================

.. py:method:: mindspore.Tensor.gather_nd(indices)

    按索引从输入Tensor中获取切片。
    使用给定的索引从具有指定形状的输入Tensor中搜集切片。
    输入Tensor的shape是 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。下文中的 `input_x` 代指输入Tensor本身。
    `indices` 是一个K维的整数张量。假定它的K-1维张量中的每一个元素是输入Tensor的切片，那么有：

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    `indices` 的最后一维不能超过输入Tensor的秩：
    :math:`indices.shape[-1] <= input\_x.rank`。

    参数：
        - **indices** (Tensor) - 获取收集元素的索引张量，其数据类型包括：int32，int64。

    返回：
        Tensor，具有与输入Tensor相同的数据类型，shape维度为 :math:`indices\_shape[:-1] + input\_x\_shape[indices\_shape[-1]:]`。

    异常：
        - **ValueError** - 如果输入Tensor的shape长度小于 `indices` 的最后一个维度。