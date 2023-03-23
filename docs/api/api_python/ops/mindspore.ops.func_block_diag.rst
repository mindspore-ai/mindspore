mindspore.ops.block_diag
=========================

.. py:function:: mindspore.ops.block_diag(inputs)

    基于输入Tensor创建块对角矩阵。

    参数：
        - **inputs** (Tensor) - 输入为一个或者多个Tensors，Tensor的维度应该为0、1或2。

    返回：
        Tensor，二维矩阵。所有输入Tensor按顺序排列，使其左上角和右下角对角线相邻，其他所有元素都置零。

    异常：
        - **TypeError** - 输入不是Tensor。
        - **ValueError** - 输入Tensor维度不为0、1或2。
