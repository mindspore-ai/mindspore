mindspore.dataset.utils.LineReader
==================================

.. py:class:: mindspore.dataset.utils.LineReader(filename)

    高效的（基于行的）文件读取。

    该类缓存基于行的文件元信息，可以让用户方便的获取文件总行数、读取文件指定行内容等。

    该类提供如下方法：
    - len() - 返回文件的总行数。
    - readline(line) - 打开文件并读取文件的第line行。
    - close() - 关闭文件句柄。

    参数：
        - **filename** (str) - 基于行的文件名。

    异常：
        - **TypeError** - `filename` 无效。
        - **RuntimeError** - `filename` 不存在或者不是普通文件。

    .. py:method:: close()

        关闭文件。

    .. py:method:: len()

        获取文件总行数。

    .. py:method:: readline(line)

        读取指定行内容。

        参数：
            - **line** (int) - 指定行号。

        返回：
            str，一行的内容，包括换行符。

        异常：
            - **TypeError** - 参数 `line` 类型错误。
            - **ValueError** - 参数 `line` 取值越界。
