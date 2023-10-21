mindspore.dataset.transforms.Slice
==================================

.. py:class:: mindspore.dataset.transforms.Slice(*slices)

    对输入进行切片。

    当前仅支持 1 维输入。

    参数：
        - **slices** (Union[int, list[int], slice, Ellipsis]) - 想要切片的片段。
          若输入类型为int，将切片指定索引值的元素，支持负索引。
          若输入类型为list[int], 将切片所有指定索引值的元素，支持负索引。
          若输入类型为 `slice <https://docs.python.org/zh-cn/3.7/library/functions.html#slice>`_ ，
          将根据其指定的起始位置、结束位置和步长进行切片。
          若输入类型为 `Ellipsis <https://docs.python.org/zh-cn/3.7/library/constants.html#Ellipsis>`_ ，即省略符，
          将切片所有元素。
          若输入为None，将切片所有元素。

    异常：      
        - **TypeError** - 当 `slices` 不为Union[int, list[int], slice, Ellipsis]类型。
