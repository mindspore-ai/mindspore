mindspore.nn.Unflatten
=======================

.. py:class:: mindspore.nn.Unflatten(axis, unflattened_size)

    根据 `axis` 和 `unflattened_size` 折叠指定维度为给定形状。
    
    参数：
        - **axis** (int) - 指定输入Tensor被折叠维度。
        - **unflattened_size** (Union(tuple[int], list[int])) - 指定维度维度折叠后的新shape，可以为tuple[int]或者list[int]。 `unflattened_size` 中各元素的乘积必须等于input_shape[axis]。

    输入：
        - **input** (Tensor) - 进行折叠操作的Tensor。

    输出：
        折叠操作后的Tensor。

        - :math:`out\_depth = ksize\_row * ksize\_col * in\_depth`
        - :math:`out\_row = (in\_row - (ksize\_row + (ksize\_row - 1) * (rate\_row - 1))) // stride\_row + 1`
        - :math:`out\_col = (in\_col - (ksize\_col + (ksize\_col - 1) * (rate\_col - 1))) // stride\_col + 1`

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `unflattened_size` 既不是tuple[int]也不是list[int]。
        - **TypeError** - `unflattened_size` 中各元素的乘积不等于input_shape[axis]。
