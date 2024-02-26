mindspore.Layout
================

.. py:class:: mindspore.Layout(device_matrix, alias_name)

    Layout描述了详细的分片信息。

    .. note::
        - 仅在半自动并行或自动并行模式下有效。

    参数：
        - **device_matrix** (tuple) - 描述设备排列的形状，其元素类型为int。
        - **alias_name** (tuple) - device_matrix的每个轴的别名，其元素类型为字符串。

    .. py:method:: to_dict

        将Layout转换为词典 。
