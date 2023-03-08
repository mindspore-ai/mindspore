mindspore.nn.Flatten
====================

.. py:class:: mindspore.nn.Flatten(start_dim=1, end_dim=-1)

    沿着从 `start_dim` 到 `end_dim` 的维度，对输入Tensor进行展平。

    参数：
        - **start_dim** (int, 可选) - 要展平的第一个维度。默认值：1。
        - **end_dim** (int, 可选) - 要展平的最后一个维度。默认值：-1。

    输入：
        - **x** (Tensor) - 要展平的输入Tensor。

    输出：
        Tensor。如果没有维度被展平，返回原始的输入 `input`，否则返回展平后的Tensor。如果 `input` 是零维Tensor，将会返回一个一维Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `start_dim` 或 `end_dim` 不是int类型。
        - **ValueError** - 规范化后，`start_dim` 大于 `end_dim`。
        - **ValueError** - `start_dim` 或 `end_dim` 不在 [-input.dim, input.dim-1] 范围内。
