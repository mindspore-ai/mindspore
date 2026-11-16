mindspore.Tensor.median
=======================

.. py:method:: mindspore.Tensor.median(axis=-1, keepdims=False) -> tuple[Tensor]

    输出Tensor指定维度上的中值与索引。

    .. warning::
        - 如果 `input` 的中值不为唯一，则 `indices` 不一定对应第一个出现的中值。该接口的具体实现方式和后端类型相关，CPU和GPU的返回值可能不相同。
        - 2.9.0（不含）之后版本非兼容性变更：删除 ``median(axis=-1, keepdims=False)`` 重载，仅支持 ``median(dim=None, keepdim=False) -> (Tensor, LongTensor)``
          （ `dim` 的默认值由 ``-1`` 改为 ``None`` ）。

    参数：
        - **axis** (int，可选) - 指定计算的轴。默认值： ``-1`` 。
        - **keepdims** (bool，可选) - 输出Tensor是否需要保留 `axis` 的维度。默认值： ``False`` 。

    返回：
        - **y** (Tensor) - 返回指定维度上的中值。数据类型与 `input` 相同。
        - **indices** (Tensor) - 指定中值索引。数据类型为int64。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `keepdims` 不是bool。
        - **ValueError** - `axis` 的范围不在[-self.dim, self.dim-1]。

    .. py:method:: mindspore.Tensor.median() -> Tensor
        :noindex:

    返回输入的中位数。

    返回：
        - **y** (Tensor) - 输出中值。

    .. py:method:: mindspore.Tensor.median(dim=-1, keepdim=False) -> tuple[Tensor]
        :noindex:

    输出指定维度 `dim` 上的中值与其对应的索引。如果 `dim` 为None，则计算Tensor中所有元素的中值。

    参数：
        - **dim** (int, 可选) - 指定计算的轴。默认值： ``None`` 。
        - **keepdim** (bool, 可选) - 是否保留 ``dim`` 指定的维度。默认值： ``False`` 。

    返回：
        - **y** (Tensor) - 输出中值，数据类型与 ``input`` 相同。

          - 如果 ``dim`` 为 ``None`` ， ``y`` 只有一个元素。
          - 如果 ``keepdim`` 为 ``True`` ， ``y`` 的shape除了在 ``dim`` 维度上为1外，与 ``input`` 一致。
          - 其他情况下， ``y`` 比 ``input`` 缺少 ``dim`` 指定的维度。

        - **indices** (Tensor) - 中值的索引。shape与 ``y`` 一致，数据类型为int64。

    异常：
        - **TypeError** - ``dim`` 不是int。
        - **TypeError** - ``keepdim`` 不是bool值。
        - **ValueError** - ``dim`` 不在 [-x.dim, x.dim-1] 范围内。
