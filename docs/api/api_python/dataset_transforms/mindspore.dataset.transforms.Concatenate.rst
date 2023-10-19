mindspore.dataset.transforms.Concatenate
========================================

.. py:class:: mindspore.dataset.transforms.Concatenate(axis=0, prepend=None, append=None)

    在输入数据的某一个轴上进行数组拼接，目前仅支持拼接形状为1D的数组。

    参数：
        - **axis** (int, 可选) - 指定一个轴用于拼接数组。默认值： ``0`` 。
        - **prepend** (numpy.ndarray, 可选) - 想要拼接到输入数据首部的数组。默认值： ``None`` ，不在首部拼接数组。
        - **append** (numpy.ndarray, 可选) - 想要拼接到输入数据尾部的数组。默认值： ``None`` ，不在尾部拼接数组。

    异常：
        - **TypeError** - 参数 `axis` 的类型不为int。
        - **TypeError** - 参数 `prepend` 的类型不为 `numpy.ndarray` 。
        - **TypeError** - 参数 `append` 的类型不为 `numpy.ndarray` 。
