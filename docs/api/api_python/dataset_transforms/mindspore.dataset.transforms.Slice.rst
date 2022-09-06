mindspore.dataset.transforms.Slice
==================================

.. py:class:: mindspore.dataset.transforms.Slice(*slices)

    使用给定的slices，对Tensor进行切片操作。功能类似于NumPy的索引(目前只支持1D形状的Tensor)。

    参数：
        - **slices** (Union[int, list[int], slice, None, Ellipsis]) - 指定切片的信息，可以为

          - 1. :py:obj:`int`: 沿着第一个维度切片对索引进行切片，支持负索引。
          - 2. :py:obj:`list(int)`: 沿着第一个维度切片所有索引进行切片，支持负号索引。
          - 3. :py:obj:`slice`: 沿着第一个维度对 `slice <https://docs.python.org/zh-cn/3.7/library/functions.html?highlight=slice#slice>`_ 对象生成的索引进行切片。
          - 4. :py:obj:`None`: 切片整个维度，类似于Python索引中的语法 :py:obj:`[:]` 。
          - 5. :py:obj:`Ellipsis`: 切片整个维度，效果与 `None` 相同。

    异常：      
        - **TypeError** - 参数 `slices` 类型不为int、list[int]、:py:obj:`slice` 、:py:obj:`None` 或 :py:obj:`Ellipsis` 。