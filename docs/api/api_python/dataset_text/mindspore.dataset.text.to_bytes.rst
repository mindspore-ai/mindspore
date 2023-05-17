mindspore.dataset.text.to_bytes
================================

.. py:function:: mindspore.dataset.text.to_bytes(array, encoding='utf8')

    基于 `encoding` 字符集对每个元素进行编码，将 `string` 的NumPy数组转换为 `bytes` 的数组。

    参数：
        - **array** (numpy.ndarray) - 表示 `string` 类型的数组，代表字符串。
        - **encoding** (str) - 表示用于编码的字符集。默认值： ``'utf8'`` 。

    返回：
        numpy.ndarray，表示 `bytes` 的NumPy数组。
