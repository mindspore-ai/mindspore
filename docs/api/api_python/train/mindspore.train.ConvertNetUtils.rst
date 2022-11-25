mindspore.train.ConvertNetUtils
================================

.. py:class:: mindspore.train.ConvertNetUtils

    将网络转换为thor层网络，用于计算并存储二阶信息矩阵。

    .. py:method:: convert_to_thor_net(net)

        该接口用于将网络转换为thor层网络，用于计算并存储二阶信息矩阵。

        .. note::
            此接口由二阶优化器thor自动调用。

        参数：
            - **net** (Cell) - 由二阶优化器thor训练的网络。
