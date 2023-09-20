mindspore.dataset.utils.LineReader
==================================

.. py:class:: mindspore.dataset.utils.LineReader(filename)

    基于行的文件读取器。

    通过提前缓存文件基于行的元信息，实现对文件各行的随机访问读取。

    参数：
        - **filename** (str) - 待读取的文件名。

    异常：
        - **TypeError** - `filename` 无效。
        - **RuntimeError** - `filename` 不存在或者不是常规文件。

    .. py:method:: close()

        关闭文件句柄。

    .. py:method:: len()

        获取当前文件的总行数。

    .. py:method:: readline(line)

        读取指定行的内容。

        参数：
            - **line** (int) - 待读取的行号，起始行号为1。

        返回：
            str，对应行的内容，不包含换行符。

        异常：
            - **TypeError** - 如果 `line` 不是int类型。
            - **ValueError** - 如果 `line` 取值大于文件总行数。
