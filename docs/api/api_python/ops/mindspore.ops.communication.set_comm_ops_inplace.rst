mindspore.ops.communication.set_comm_ops_inplace
==================================================

.. py:function:: mindspore.ops.communication.set_comm_ops_inplace(is_enable, func_list=None)

    设置通信函数的inplace属性。

    参数：
        - **is_enable** (bool) - 是否启用inplace。
        - **func_list** (list) - 指示哪些函数设置了inplace属性。

    异常：
        - **TypeError** - 如果 `is_enable` 不是bool。
        - **TypeError** - 如果 `func_list` 不是None且不是list。
        - **ValueError** - `func_list` 中的函数名无效。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
