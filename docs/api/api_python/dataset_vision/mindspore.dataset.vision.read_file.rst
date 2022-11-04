mindspore.dataset.vision.read_file
==================================

.. py:function:: mindspore.dataset.vision.read_file(filename)

    以二进制模式读取文件。

    参数：
        - **filename** (str) - 待读取文件路径。

    返回：
        - numpy.ndarray, 一维uint8类型数据。
        
    异常：
        - **TypeError** - 如果 `filename` 不是str类型。
        - **RuntimeError** - 如果 `filename` 不存在或不是普通文件。
