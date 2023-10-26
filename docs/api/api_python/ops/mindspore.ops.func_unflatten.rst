mindspore.ops.unflatten
=======================

.. py:function:: mindspore.ops.unflatten(input, axis, unflattened_size)

    折叠输入Tensor `input` 的指定维度 `axis` ，返回shape为 `unflattened_size` 的Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (int) - 指定输入Tensor被折叠维度。
        - **unflattened_size** (Union(tuple[int], list[int])) - 指定维度折叠后的新shape。 `unflattened_size` 中各元素的乘积必须等于input_shape[axis]。

    返回：
        折叠操作后的Tensor。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `unflattened_size` 既不是tuple[int]也不是list[int]。
        - **TypeError** - `unflattened_size` 中各元素的乘积不等于input_shape[axis]。