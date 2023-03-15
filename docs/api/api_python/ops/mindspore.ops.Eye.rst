mindspore.ops.Eye
==================

.. py:class:: mindspore.ops.Eye

    创建一个主对角线上元素为1，其余元素为0的Tensor。

    更多参考详见 :func:`mindspore.ops.eye`。

    输入：
        - **n** (int) - 指定返回Tensor的行数。仅支持常量值。
        - **m** (int) - 指定返回Tensor的列数。仅支持常量值。
        - **t** (mindspore.dtype) - 指定返回Tensor的数据类型。数据类型是 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。

    输出：
        Tensor，主对角线上为1，其余的元素为0。它的shape由 `n` 和 `m` 指定。数据类型由 `t` 指定。
