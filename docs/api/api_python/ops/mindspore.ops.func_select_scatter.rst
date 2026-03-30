mindspore.ops.select_scatter
============================

.. py:function:: mindspore.ops.select_scatter(input, src, axis, index)

    将 `src` 中的值散布到 `input` 指定轴 `axis` 的指定位置 `index` 上。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `axis` 将重命名为 `dim` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **src** (Tensor) - 源tensor。
        - **axis** (int) - 指定轴。
        - **index** (int) - 在指定轴上散布的位置。

    返回：
        Tensor
