mindspore.ops.tensor_split
===========================

.. py:function:: mindspore.ops.tensor_split(input, indices_or_sections, axis=0)

    根据指定的轴将输入Tensor进行分割成多个子Tensor。

    参数：
        - **input** (Tensor) - 待分割的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 

          - 如果 `indices_or_sections` 是整数类型n，输入tensor将分割成n份。

            - 如果 :math:`input.size(axis)` 能被n整除，那么子切片的大小相同，为 :math:`input.size(axis) / n` 。
            - 如果 :math:`input.size(axis)` 不能被n整除，那么前 :math:`input.size(axis) % n` 个切片的大小为 :math:`input.size(axis) // n + 1` ，其余切片的大小为 :math:`input.size(axis) // n` 。

          - 如果 `indices_or_sections` 类型为tuple(int) 或 list(int)，那么输入tensor将在tuple或list中的索引处切分。例如：给定参数 :math:`indices\_or\_sections=[1, 4]` 和 :math:`axis=0` 将得到切片 :math:`input[:1]` ， :math:`input[1:4]` ，和 :math:`input[4:]` 。
        - **axis** (int) - 指定分割轴。默认值：0。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - 参数 `axis` 超出 :math:`[-input.dim, input.dim)` 范围。
        - **TypeError** - `indices_or_sections` 中的每个元素不是int类型
        - **TypeError** - `indices_or_sections` 不是int，tuple(int)或list(int)。


