mindspore.get_grad
==================

.. py:function:: mindspore.get_grad(gradients, identifier)

    当 :func:`mindspore.grad` 的 `return_ids` 参数设置为True时，将其返回值作为 `gradients` 。再根据 `identifier` 在 `gradients` 中找到对应的梯度值。

    根据 `identifier` 查找梯度值包含以下两种场景：

    1. `identifier` 为指定求导输入位置的索引；
    2. `identifier` 为网络变量。

    参数：
        - **gradients** (Union[tuple[int, Tensor], tuple[tuple, tuple]]) - :func:`mindspore.grad` 参数 `return_ids` 为True时的返回值。
        - **identifier** (Union[int, Parameter]) - 指定求导输入位置的索引或者网络变量。

    返回：
        `identifier` 所指定的求导输入位置的索引所对应的Tensor的梯度值，或者网络变量所对应的Tensor的梯度值。

    异常：
        - **RuntimeError** - 无法找到identifier所对应的梯度值。
        - **TypeError** - 入参类型不符合要求。
