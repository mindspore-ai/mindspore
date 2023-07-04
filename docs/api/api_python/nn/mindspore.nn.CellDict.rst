mindspore.nn.CellDict
=====================

.. py:class:: mindspore.nn.CellDict(*args, **kwargs)

    构造Cell字典。关于 `Cell` 的介绍，可参考 :class:`mindspore.nn.Cell` 。

    `CellDict` 可以像普通Python字典一样使用。

    参数：
        - **args** (iterable，可选) - 一个键值对类型为(string, Cell)的可迭代对象，或者是一个从string到Cell的映射（字典）。通过类型为string的键可以在CellDict中查找其对应的Cell。
        - **kwargs** (dict) - 为待扩展的关键字参数预留。

    .. py:method:: clear()

        移除CellDict中的所有Cell。

    .. py:method:: items()

        返回包含CellDict中所有键值对的可迭代对象。

    .. py:method:: keys()

        返回包含CellDict中所有键的可迭代对象。

    .. py:method:: pop(key)

        从CellDict中移除键为key的Cell，并将这个Cell返回。

        参数：
            - **key** (string) - 从CellDict中移除的Cell的键。

        异常：
            - **TypeError** - 如果key不是string类型。
            - **KeyError** - key对应的Cell在CellDict中不存在。

    .. py:method:: update(cells)

        使用映射或者可迭代对象中的键值对来更新CellDict中已存在的Cell。

        参数：
            - **cells** (iterable) - 一个键值对类型为(string, Cell)的可迭代对象，或者是一个从string到Cell的映射（字典）。

        .. note::
            如果 `cells` 是一个CellDict、一个OrderedDict或者是一个包含键值对的可迭代对象，那么新增元素的顺序在CellDict中仍会被保留。

    .. py:method:: values()

        返回包含CellDict中所有Cell的可迭代对象。
