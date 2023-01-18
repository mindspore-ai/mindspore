mindspore.QuantDtype
====================

.. py:class:: mindspore.QuantDtype

    MindSpore量化数据类型枚举类，包含 `INT1` ~ `INT16`，`UINT1` ~ `UINT16` 。

    `QuantDtype` 定义在 `/mindspore/common/dtype.py` 文件下 。运行以下命令导入环境：

    .. code-block::

        from mindspore import QuantDtype

    .. py:method:: value()

        获取当前 `QuantDtype` 的值。

        返回：
            int，表示当前 `QuantDtype` 的值。
