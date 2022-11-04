mindspore.dataset.vision.write_file
===================================

.. py:function:: mindspore.dataset.vision.write_file(filename, data)

    使用二进制模式将一维uint8类型数据数组写到文件。

    参数：
        - **filename** (str) - 要写入的文件的路径。
        - **data** (Union[numpy.ndarray, mindspore.Tensor]) - 要写入的一维uint8数据。

    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **TypeError** - 如果 `data` 不是numpy.ndarray或mindspore.Tensor类型。
        - **RuntimeError** - 如果 `filename` 不是普通文件。
        - **RuntimeError** - 如果 `data` 的数据类型不是uint8类型。
        - **RuntimeError** - 如果 `data` 的shape不是一维数组。
