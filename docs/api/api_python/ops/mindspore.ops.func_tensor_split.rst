mindspore.ops.tensor_split
===========================

.. py:function:: mindspore.ops.tensor_split(x, indices_or_sections, axis=0)

    根据指定的轴将输入Tensor进行分割成多个子Tensor。

    参数：
        - **x** (Tensor) - 待分割的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 如果 `indices_or_sections` 是整数类型n，输入将沿 `axis` 轴分割成n份。如果输入沿着 `axis` 轴能被n整除，那么每个切片的大小相同为 :math:`x.size(axis) / n` 。如果不能被n整除，那么前 :math:`x.size(axis) % n` 个切片的大小为 :math:`x.size(axis) // n + 1` ，其余切片的大小为 :math:`x.size(axis) // n` 。
          如果 `indices_or_sections` 是由int组成list或者tuple，那么输入将沿着 `axis` 轴在tuple或list中的索引处切分。例如：:math:`indices\_or\_sections=[2, 3]` 和 :math:`axis=0` 将得到切片 :math:`x[:2]` ， :math:`x[2:3]` ，和 :math:`x[3:]` .
        - **axis** (int) - 指定分割轴。默认值：0。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - 参数 `axis` 超出 :math:`[-x.dim, x.dim)` 范围。
        - **TypeError** - `indices_or_sections` 中的每个元素不是int类型
        - **TypeError** - `indices_or_sections` 不是int，tuple(int)或list(int)。


