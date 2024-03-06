mindspore.parameter_broadcast
======================================

.. py::function:: mindspore.parameter_broadcast(net, layout, cur_rank=0, initial_rank=0)

    在数据并行维度将参数广播给另外的卡。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **net** (Cell) - 参数将被广播的网络。
        - **layout** (Dict) - 参数排布字典。 来自 :meth:`mindspore.nn.Cell.parameter_layout_dict` 或
          从文件中读取(如: "strategy.ckpt")。key为参数名， value为该参数的layout。
        - **cur_rank** (int, 可选) - 当前卡的rank id。默认值: ``0``。
        - **initial_rank** (int， 可选) - 当前流水线并行stage起始rank id。默认值: ``0``。

    异常:
        - **ValueError** - `cur_rank` 是错的。
        - **ValueError** - `initial_rank` 是错的。
        - **ValueError** - `layout` 中的参数名在 :meth:`mindspore.nn.Cell.parameters_dict` 中找不到。
