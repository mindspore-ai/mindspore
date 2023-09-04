mindspore.amp.auto_mixed_precision
==================================

.. py:function:: mindspore.amp.auto_mixed_precision(network, amp_level="O0", dtype=mstype.float16)

    返回一个经过自动混合精度处理的网络。

    该接口会对输入网络进行自动混合精度处理，处理后的网络里的Cell和算子增加了精度转换操作，以低精度进行计算，如 ``mstype.float16`` 或 ``mstype.bfloat16`` 。
    Cell和算子的输入和参数被转换成低精度浮点数，计算结果被转换回全精度浮点数，即  ``mstype.float32`` 。

    框架内置了一组黑名单和白名单， `amp_level` 决定了具体对哪些Cell和算子进行精度转换。

    当前的内置白名单内容为：

    [:class:`mindspore.nn.Conv1d`, :class:`mindspore.nn.Conv2d`, :class:`mindspore.nn.Conv3d`,
    :class:`mindspore.nn.Conv1dTranspose`, :class:`mindspore.nn.Conv2dTranspose`,
    :class:`mindspore.nn.Conv3dTranspose`, :class:`mindspore.nn.Dense`, :class:`mindspore.nn.LSTMCell`,
    :class:`mindspore.nn.RNNCell`, :class:`mindspore.nn.GRUCell`, :class:`mindspore.ops.Conv2D`,
    :class:`mindspore.ops.Conv3D`, :class:`mindspore.ops.Conv2DTranspose`,
    :class:`mindspore.ops.Conv3DTranspose`, :class:`mindspore.ops.MatMul`, :class:`mindspore.ops.BatchMatMul`,
    :class:`mindspore.ops.PReLU`, :class:`mindspore.ops.ReLU`, :class:`mindspore.ops.Ger`]

    当前的内置黑名单内容为：

    [:class:`mindspore.nn.BatchNorm1d`, :class:`mindspore.nn.BatchNorm2d`, :class:`mindspore.nn.BatchNorm3d`,
    :class:`mindspore.nn.LayerNorm`]

    关于自动混合精度的详细介绍，请参考 `自动混合精度 <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html>`_ 。

    .. note::
        - 重复调用混合精度接口，如 `custom_mixed_precision` 和 `auto_mixed_precision` ，可能导致网络层数增大，性能降低。
        - 如果使用 :class:`mindspore.train.Model` 和 :func:`mindspore.amp.build_train_network` 等接口来训练经
          过 `custom_mixed_precision` 和 `auto_mixed_precision` 等混合精度接口转换后的网络，则需要将 `amp_level` 配置
          为 ``O0`` 以避免重复的精度转换。

    参数：
        - **network** (Cell) - 定义网络结构。
        - **amp_level** (str) - 支持["O0", "O1", "O2", "O3"]。默认值： ``"O0"`` 。

          - **"O0"** - 不变化。
          - **"O1"** - 仅将白名单内的Cell和算子转换为低精度运算，其余部分保持全精度运算。
          - **"O2"** - 黑名单内的Cell和算子保持全精度运算，其余部分都转换为低精度运算。
          - **"O3"** - 将网络全部转为低精度运算。

        - **dtype** (Type) - 低精度计算时使用的数据类型，可以是 ``mstype.float16`` 或 ``mstype.bfloat16`` 。默认值： ``mstype.float16`` 。

    异常：
        - **TypeError** - `network` 不是Cell。
        - **ValueError** - `amp_level` 不在支持范围内。
        - **ValueError** - `dtype` 既不是 ``mstype.float16`` 也不是 ``mstype.bfloat16`` 。
