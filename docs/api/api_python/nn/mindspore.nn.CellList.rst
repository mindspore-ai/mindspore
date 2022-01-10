mindspore.nn.CellList
======================

.. py:class:: mindspore.nn.CellList(*args, **kwargs)

    构造Cell列表。关于Cell的介绍，可参考 `Cell <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_。

    CellList可以像普通Python列表一样使用，其包含的Cell均已初始化。

    **参数：**

    **args** (list，可选) - Cell列表。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> import mindspore.nn as nn
    >>>
    >>> conv = nn.Conv2d(100, 20, 3)
    >>> bn = nn.BatchNorm2d(20)
    >>> relu = nn.ReLU()
    >>> cell_ls = nn.CellList([bn])
    >>> cell_ls.insert(0, conv)
    >>> cell_ls.append(relu)
    >>> cell_ls.extend([relu, relu])

    .. py:method:: append(cell)

        在列表末尾添加一个Cell。

        **参数：**

        **cell** (Cell) - 要添加的Cell。

    .. py:method:: extend(cells)

        将cells中的Cell添加到列表末尾。

        **参数：**

        **cells** (list) - 要添加的Cell列表。

        **异常：**

        **TypeError** - cells中的元素不是Cell。

    .. py:method:: insert(index, cell)

        在列表中的给定索引之前插入给定的Cell。

        **参数：**

        - **index** (int) - 给定的列表索引。
        - **cell** (Cell) - 要插入的Cell。