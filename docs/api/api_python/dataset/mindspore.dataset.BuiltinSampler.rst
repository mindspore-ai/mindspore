.. py:method:: add_child(sampler)

    为给定采样器添加子采样器。父采样器接收子采样器输出数据作为输入，并应用其采样逻辑返回新的采样结果。

    .. note::
        被添加的子sampler如果有 `shuffle` 属性，其值不能是 ``Shuffle.PARTIAL`` ，且父sampler的 `shuffle` 属性值必须是 ``Shuffle.GLOBAL`` 。

    参数：
        - **sampler** (Sampler) - 用于从数据集中选择样本的对象。仅支持内置采样器（ :class:`mindspore.dataset.DistributedSampler` 、 :class:`mindspore.dataset.RandomSampler` 、 :class:`mindspore.dataset.SequentialSampler` ）。

.. py:method:: get_child()

    获取给定采样器的子采样器。

    返回：
        Sampler，给定采样器的子采样器。
