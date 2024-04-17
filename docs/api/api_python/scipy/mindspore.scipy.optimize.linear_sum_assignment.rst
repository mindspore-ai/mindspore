mindspore.scipy.optimize.linear_sum_assignment
==============================================

.. py:function:: mindspore.scipy.optimize.linear_sum_assignment(cost_matrix, maximize, dimension_limit=Tensor(sys.maxsize))

    解决线性和分配问题（指派问题）。

    指派问题如下表示：

    .. math::
        min\sum_{i}^{} \sum_{j}^{} C_{i,j} X_{i,j}

    其中， :math:`C` 表示代价矩阵， :math:`X_{i,j} = 1` 表示列 :math:`j` 被指派给了行 :math:`i` 。

    参数：
        - **cost_matrix** (Tensor) - 二维代价矩阵。其shape为 :math:`(M, N)` 。
        - **maximize** (bool) - 为True则计算最大权值匹配，否则计算最小权值匹配。
        - **dimension_limit** (Tensor，可选) - 用来限制 ``cost_matrix`` 第二维的实际大小。默认值： ``Tensor(sys.maxsize)`` ，表示没有限制。类型为零维int64类型Tensor。

    返回：
        由 `row_idx` 和 `col_idx` 组成的tuple。

        - **row_idx** (Tensor) - 指派问题的行索引。如果指定了 `dimension_limit` ，用-1在结尾补齐。其shape为 :math:`(N, )` , 其中 :math:`N` 为 `cost_matrix` 维度较小值。
        - **col_idx** (Tensor) - 指派问题的列索引。如果指定了 `dimension_limit` ，用-1在结尾补齐。其shape为 :math:`(N, )` , 其中 :math:`N` 为 `cost_matrix` 维度较小值。

    异常：
        - **TypeError** - 如果 `cost_matrix` 数据类型不是float16、float32、float64、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool之一。
        - **TypeError** - 如果 `maximize` 的类型不是bool。
        - **TypeError** - 如果 `dimension_limit` 的数据类型不是int64。
        - **ValueError** - 如果 `cost_matrix` 维度不为2。