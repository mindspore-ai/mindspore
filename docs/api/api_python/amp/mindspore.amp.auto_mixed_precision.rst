mindspore.amp.auto_mixed_precision
==================================

.. py:function:: mindspore.amp.auto_mixed_precision(network, amp_level="O0")

    返回一个经过自动混合精度处理的网络。

    该接口会对输入网络进行自动混合精度处理，处理后的网络里的Cell和算子增加了精度转换操作，以float16精度进行计算。
    Cell和算子的输入和参数被转换成float16类型，计算结果被转换回float32类型。

    框架内置了一组黑名单和白名单， `amp_level` 决定了具体对哪些Cell和算子进行精度转换：

    - 当 `amp_level="O0"` 时，不进行精度转换。
    - 当 `amp_level="O1"` 时，仅将白名单内的Cell和算子进行精度转换。
    - 当 `amp_level="O2"` 时，将除了黑名单内的其他Cell和算子都进行精度转换。
    - 当 `amp_level="O3"` 时，将网络里的所有Cell和算子都进行精度转换。

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

    参数：
        - **network** (Cell) - 定义网络结构。
        - **amp_level** (str) - 支持["O0", "O1", "O2", "O3"]。默认值： ``"O0"`` 。

          - **"O0"** - 不变化。
          - **"O1"** - 将白名单内的Cell和算子转换为float16精度运算，其余部分保持float32精度运算。
          - **"O2"** - 将黑名单内的Cell和算子保持float32精度运算，其余部分转换为float16精度运算。
          - **"O3"** - 将网络全部转为float16精度。

    异常：
        - **ValueError** - `amp_level` 不在支持范围内。
