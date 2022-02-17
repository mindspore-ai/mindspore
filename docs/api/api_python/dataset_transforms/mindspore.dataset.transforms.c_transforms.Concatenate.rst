mindspore.dataset.transforms.c_transforms.Concatenate
=====================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.Concatenate(axis=0, prepend=None, append=None)

    在Tensor的某一个轴上进行元素拼接。

    **参数：**

    - **axis** (int, 可选) - 指定一个轴用于拼接Tensor，默认值：0。
    - **prepend** (numpy.array, 可选) - 指定拼接在最前面的Tensor，默认值：None，不指定。
    - **append** (numpy.array, 可选) - 指定拼接在最后面的Tensor，默认值：None，不指定。

    **异常：**
      
    - **TypeError** - 参数 `axis` 的类型不为 int。
    - **TypeError** - 参数 `prepend` 的类型不为 numpy.array。
    - **TypeError** - 参数 `append` 的类型不为 numpy.array。
