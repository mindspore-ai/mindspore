mindspore.mint.distributed.is_available
=============================================

.. py:function:: mindspore.mint.distributed.is_available()

    分布式模块是否可用。

    .. note::
        - MindSpore在所有平台下都支持分布式，因此该函数返回值恒为True。
        - 当前仅支持PyNative模式，不支持Graph模式。

    返回：
        bool，表示分布式模块是否可用。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
