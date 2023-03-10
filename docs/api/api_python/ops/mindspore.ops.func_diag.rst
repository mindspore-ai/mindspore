mindspore.ops.diag
===================

.. py:function:: mindspore.ops.diag(input)

    用给定的对角线值构造对角线Tensor。

    假设输入Tensor维度为 :math:`[D_1,... D_k]` ，则输出是一个rank为2k的tensor，其维度为 :math:`[D_1,..., D_k, D_1,..., D_k]` ，其中 :math:`output[i_1,..., i_k, i_1,..., i_k] = input[i_1,..., i_k]` 并且其他位置的值为0。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，具有与输入Tensor相同的数据类型。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的rank小于1。
