mindspore.dataset.text.to_str
==============================

.. py:function:: mindspore.dataset.text.to_str(array, encoding='utf8')

    基于 `encoding` 字符集对每个元素进行解码，借此将 `bytes` 的NumPy数组转换为 `string` 的数组。

    参数：
        - **array** (numpy.ndarray) - 表示 `bytes` 类型的数组，代表字符串。
        - **encoding** (str) - 表示用于解码的字符集。默认值： ``'utf8'`` 。

    返回：
        numpy.ndarray，表示 `str` 的NumPy数组。
