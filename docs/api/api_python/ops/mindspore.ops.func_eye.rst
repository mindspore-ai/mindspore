mindspore.ops.eye
==================

.. py:function:: mindspore.ops.eye(n, m=None, dtype=None)

    创建一个主对角线上元素为1，其余元素为0的Tensor。

    .. note::
        结合ReverseV2算子可以得到一个反对角线为1的Tensor，但是目前ReverseV2算子只支持Ascend和GPU平台。

    参数：
        - **n** (int) - 指定返回Tensor的行数。仅支持常量值。
        - **m** (int) - 指定返回Tensor的列数。仅支持常量值。默认值为None，返回Tensor的列数默认与n相等。
        - **dtype** (mindspore.dtype) - 指定返回Tensor的数据类型。数据类型必须是 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。默认值为None，返回Tensor的数据类型默认为mindspore.float32。

    返回：
        Tensor，主对角线上为1，其余的元素为0。它的shape由 `n` 和 `m` 指定。数据类型由 `dtype` 指定。

    异常：
        - **TypeError** - `m` 或 `n` 不是int。
        - **ValueError** - `m` 或 `n` 小于1。
