mindspore.amp.custom_mixed_precision
=====================================

.. py:function:: mindspore.amp.custom_mixed_precision(network, *, white_list=None, black_list=None)

    通过配置白名单或黑名单，对Cell进行自定义混合精度处理。
    当提供 `white_list` 时，网络中包含在 `white_list` 里的Primitive或Cell会进行精度转换。
    当提供 `black_list` 时，网络中不包含在 `black_list` 里的Primitive或Cell会进行精度转换。
    需要提供 `white_list` 和 `black_list` 中的一个。

    .. note::

        - 这是一个实验性接口，后续可能删除或修改。
        - `custom_mixed_precision` 和 `auto_mixed_precision` 不应该同时使用。当需要同时使用 `build_train_network` 和 `custom_mixed_precision` 时，应该先使用 `level='O0'` 参数调用 `build_train_network` ，然后再调用 `custom_mixed_precision` 。
        - 当使用黑名单时，Primitive类型还未支持。

    参数：
        - **network** (Cell) - 定义网络结构。
        - **white_list** (list[Primitive, Cell], optional) - 自定义混合精度的白名单。默认值：None。
        - **black_list** (list[Primitive, Cell], optional) - 自定义混合精度的黑名单。默认值：None。

    异常：
        - **ValueError** - `network` 的类型不是Cell。
        - **ValueError** -  `white_list` 和 `black_list` 都没提供。
        - **ValueError** -  同时提供了 `white_list` 和 `black_list` 。

    返回：
        network (Cell)：支持混合精度的网络。
