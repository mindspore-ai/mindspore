mindspore.nn.CellList
======================

.. py:class:: mindspore.nn.CellList(*args, **kwargs)

    构造Cell列表。关于Cell的介绍，可参考 `Cell <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_。

    CellList可以像普通Python列表一样使用，其包含的Cell均已初始化。

    参数：
        - **args** (list，可选) - Cell列表。

    .. py:method:: append(cell)

        在列表末尾添加一个Cell。

        参数：
            - **cell** (Cell) - 要添加的Cell。

    .. py:method:: extend(cells)

        将cells中的Cell添加到列表末尾。

        参数：
            - **cells** (list) - 要添加的Cell列表。

        异常：
            - **TypeError** - cells中的元素不是Cell。

    .. py:method:: insert(index, cell)

        在列表中的给定索引之前插入给定的Cell。

        参数：
            - **index** (int) - 给定的列表索引。
            - **cell** (Cell) - 要插入的Cell。