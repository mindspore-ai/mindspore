mindspore.ops.Coalesce
=======================

.. py:class:: mindspore.ops.Coalesce

    返回输入的合并稀疏Tensor。

    输入：
        - **x_indices** (Tensor) - 二维Tensor，表示稀疏Tensor中非零元素的索引，所有元素的取值都是非负的。数据类型为int64。
          其shape可表示为： :math:`(y, x)` 。
        - **x_values** (Tensor) - 一维Tensor，表示与 `y_indices` 中的索引对应的值。支持的数据类型为float16、float32。
          其shape可表示为： :math:`(x,)` 。
        - **x_shape** (Tensor) - 一维Tensor，代表稀疏Tensor的shape。数据类型为int64。
          其shape可表示为： :math:`(y,)` 。

    输出：
        - **y_indices** (Tensor) - 二维Tensor，表示稀疏Tensor中非零元素的索引，所有元素的取值都是非负的。数据类型为int64。
          其shape可表示为： :math:`(y, z)` ， `z` 代表 `x_indices` 中不同索引的数量。
        - **y_values** (Tensor) - 一维Tensor，表示与 `y_indices` 中的索引对应的值。数据类型与 `x_values` 保持一致。
          其shape可表示为： :math:`(z,)` 。
        - **y_shape** (Tensor) - 一维Tensor，代表稀疏Tensor的shape。数据类型为int64。
          其shape可表示为： :math:`(y,)` 。

    异常：
        - **TypeError** - 输入 `x_values` 的数据类型不是float32或float16之一。
        - **TypeError** - 输入 `x_indices` 或 `x_shape` 中存在数据类型不是int64的取值。
        - **ValueError** -  `x_values` 或 `x_shape` 不是一维Tensor。
        - **ValueError** -  `x_indices` 不是一个二维Tensor。
        - **ValueError** -  `x_indices` 第二个维度的大小和 `x_values` 第一个维度的大小不一致。
        - **ValueError** -  `x_indices` 第一个维度的大小和 `x_shape` 第一个维度的大小不一致。
        - **ValueError** -  `x_indices` 中存在负数。
        - **ValueError** -  `x_indices` 中存在元素取值超过了 `x_shape` 的限制。