mindspore.scipy.linalg.block_diag
=================================

.. py:function:: mindspore.scipy.linalg.block_diag(*arrs)

    根据输入的数组创建块对角矩阵。

    输入为：`A`、`B` 和 `C` 的Tensor列表。输出为：在对角线上排列这些Tensor的块对角矩阵。

    .. code-block::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    .. note::
        Windows平台上还不支持 `block_diag`。

    参数：
        - **arrs** (list) - 最大支持2D的Tensor输入。
          一个或多个Tensor，维度支持0D，1D、2D。

    返回：
        对角线上含有 `A`、`B`、`C`，...的Tensor，数据类型与 `A` 相同。

    异常：
        - **ValueError** - 输入参数中存在维度大于2的Tensor。
