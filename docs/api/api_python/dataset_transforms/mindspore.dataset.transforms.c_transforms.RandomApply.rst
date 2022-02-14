mindspore.dataset.transforms.c_transforms.RandomApply
=====================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.RandomApply(transforms, prob=0.5)

    在一组数据增强中按概率应用其中的增强处理。

    **参数：**

    - **transforms** (transforms) - 一个数据增强的列表。
    - **prob** (float, 可选) - 随机应用某个数据增强的概率，默认值：0.5。

    **异常：**
      
    - **TypeError** - 参数 `transforms` 类型不为 list。
    - **ValueError** - 参数 `transforms` 的长度为空。
    - **TypeError** - 参数 `transforms` 的元素不是Python可调用对象
      或类型不为 :class:`mindspore.dataset.transforms.c_transforms.TensorOperation` 。
    - **TypeError** - 参数 `prob` 的类型不为bool。
    - **ValueError** - 参数 `prob` 的取值范围不为[0.0, 1.0]。
