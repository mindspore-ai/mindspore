mindspore.get_grad
==================

.. py:function:: mindspore.get_grad(gradients, identifier)

    根据输入的identifier，在mindspore.grad的return_ids参数为True时返回的gradients中，找到对应的梯度值的函数。

    根据identifier查找梯度值包含以下两种场景：

    1. identifier为指定求导输入位置的索引；
    2. identifier为网络变量。

    参数：
        - **gradients** (Union[tuple[int, Tensor], tuple[tuple, tuple]]) - mindspore.grad参数return_ids为True时的返回值。
        - **identifier** (Union[int, Parameter]) - 指定求导输入位置的索引或者网络变量。

    返回：
        identifier所指定的求导输入位置的索引所对应的梯度值，或者网络变量所对应的梯度值。

    异常：
        - **RuntimeError** - 无法找到identifier所对应的梯度值。
        - **TypeError** - 入参类型不符合要求。
