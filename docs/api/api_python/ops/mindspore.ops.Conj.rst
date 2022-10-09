mindspore.ops.Conj
====================

.. py:class:: mindspore.ops.Conj

    返回输入张量的共轭。
    如果输入的复数形式为a + bj，其中a是实部，b是虚部。返回值为a – bj。
    如果输入为实数，则返回值不变。
    
    输入：
        - **input** (Tensor) - 要计算到的输入张量。必须具有数字类型。
    
    输出：
        Tensor，具有与输入相同的dtype。
    
    异常：
        - **TypeError** - 如果输入的dtype不是数字类型。
        - **TypeError** - 如果输入不是Tensor。
