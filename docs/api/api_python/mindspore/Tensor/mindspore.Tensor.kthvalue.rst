mindspore.Tensor.kthvalue
=========================

.. py:method:: mindspore.Tensor.kthvalue(k, dim=-1, keepdim=False)

    计算 `self` Tensor沿指定维度 `dim` 的第 `k` 小值，并返回一个由(`values`, `indices`)组成的Tuple，其中 `values` 包含第 `k` 小的元素， `indices` 提供每个对应元素的索引。

    参数：
        - **k** (int) - 指定要检索的第 `k` 小的元素。
        - **dim** (int，可选) - 指定用于查找第 `k` 小值的维度。默认值为 ``-1`` 。
        - **keepdim** (bool，可选) - 是否保留维度。如果设置为 ``True`` ，输出将与 `self` 保持相同的维度；如果设置为 ``False`` ，输出将减少对应的维度。默认值为 ``False`` 。

    返回：
        返回一个包含 `values` 和 `indices` 的Tuple。

        - **values** (Tensor) - `self` 的第 `k` 小值，与 `self` 的数据类型相同。
        - **indices** (Tensor) -  `self` Tensor第 `k` 小值的索引，其shape与 `values` 相同，数据类型为int64。

    异常：
        - **TypeError** - 如果 `k` 或者 `dim` 不是int。
        - **TypeError** - 如果 `keepdim` 不是bool。
        - **TypeError** - 如果 `self` 的数据类型不支持。
        - **ValueError** - 如果 `self` 为空Tensor。
        - **RuntimeError** - 如果 `k` 不在有效范围内。
