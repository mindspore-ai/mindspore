mindspore.QuantDtype
====================

.. py:class:: mindspore.QuantDtype

    MindSpore量化数据类型枚举类，包含 `INT1` ~ `INT16`，`UINT1` ~ `UINT16` 。

    `QuantDtype` 定义在 `dtype.py <https://gitee.com/mindspore/mindspore/blob/master/mindspore/python/mindspore/common/dtype.py>`_ 文件下 。运行以下命令导入环境：

    .. code-block::

        from mindspore import QuantDtype

    教程样例：
        - `昇思金箍棒量化感知训练时配置算法
          <https://www.mindspore.cn/golden_stick/docs/zh-CN/master/quantization/slb.html#%E5%BA%94%E7%94%A8%E9%87%8F%E5%8C%96%E7%AE%97%E6%B3%95>`_

    .. py:method:: value()

        获取当前 `QuantDtype` 的值。该接口当前主要用于序列化或反序列化 `QuantDtype` 。

        返回：
            int，表示当前 `QuantDtype` 的值。        
