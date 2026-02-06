mindspore.amp.custom_mixed_precision
=====================================

.. py:function:: mindspore.amp.custom_mixed_precision(network, *, white_list=None, black_list=None, dtype=mstype.float16)

    通过配置白名单或黑名单，对Cell进行自定义混合精度处理。

    - 当提供 `white_list` 时，网络中包含在 `white_list` 中的Primitive或Cell会进行精度转换。
    - 当提供 `black_list` 时，网络中不包含在 `black_list` 中的Cell会进行精度转换。
    
    需要提供 `white_list` 和 `black_list` 中的一个。

    .. note::
        - 重复调用混合精度接口，如 `custom_mixed_precision` 和 `auto_mixed_precision` ，可能导致网络层数增大、性能降低。
        - 如果使用 :class:`mindspore.train.Model` 和 :func:`mindspore.amp.build_train_network` 等接口来训练经\
          过 `custom_mixed_precision` 和 `auto_mixed_precision` 等混合精度接口转换后的网络，则需要\
          将 :class:`mindspore.train.Model` 和 :func:`mindspore.amp.build_train_network` 等接口的 `amp_level` 或 `level` 配置\
          为 ``O0`` ，以避免重复的精度转换。
        - 当使用黑名单时，Primitive类型还未支持。

    参数：
        - **network** (Cell) - 定义网络结构。

    关键字参数：
        - **white_list** (list[Primitive, Cell], 可选) - 自定义混合精度的白名单。默认值： ``None`` 。
        - **black_list** (list[Cell], 可选) - 自定义混合精度的黑名单。默认值： ``None`` 。
        - **dtype** (Type, 可选) - 低精度计算时使用的数据类型，可以是 ``mstype.float16`` 或 ``mstype.bfloat16``。默认值： ``mstype.float16`` 。

    返回：
        network (Cell)，支持混合精度的网络。

    异常：
        - **TypeError** - `network` 的类型不是Cell。
        - **ValueError** - `white_list` 和 `black_list` 都没提供。
        - **ValueError** - 同时提供了 `white_list` 和 `black_list` 。
        - **ValueError** - `dtype` 既不是 ``mstype.float16`` 也不是 ``mstype.bfloat16`` 。
