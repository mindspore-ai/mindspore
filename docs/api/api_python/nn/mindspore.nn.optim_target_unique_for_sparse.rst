    .. py:method:: target
        :property:

        该属性用于指定在主机（host）上还是设备（device）上更新参数。输入类型为str，只能是'CPU'，'Ascend'或'GPU'。

    .. py:method:: unique
        :property:

        该属性表示是否在优化器中进行梯度去重，通常用于稀疏网络。如果梯度是稀疏的则设置为True。如果前向稀疏网络已对权重去重，即梯度是稠密的，则设置为False。未设置时默认值为True。
