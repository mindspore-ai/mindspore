mindspore.nn.CellDict
=====================

.. py:class:: mindspore.nn.CellDict(*args, **kwargs)

    构造Cell字典。关于 `Cell` 的介绍，可参考 :class:`mindspore.nn.Cell` 。

    `CellDict` 可以像普通Python字典一样使用。

    参数：
        - **args** (iterable，可选) - 一个可迭代对象，通过它可迭代若干键值对(key, Cell)，其中键值对的类型为(string, Cell)；或者是一个从string到Cell的映射（字典）。Cell的类型不能为CellDict, CellList或者SequentialCell。key不能与类Cell中的属性重名，不能包含‘.’，不能是一个空串。通过类型为string的键可以在CellDict中查找其对应的Cell。
        - **kwargs** (dict) - 为待扩展的关键字参数预留。

    .. py:method:: clear()

        移除CellDict中的所有Cell。

    .. py:method:: items()

        返回包含CellDict中所有键值对的可迭代对象。

        返回：
            一个可迭代对象。

    .. py:method:: keys()

        返回包含CellDict中所有键的可迭代对象。

        返回：
            一个可迭代对象。

    .. py:method:: pop(key)

        从CellDict中移除键为 `key` 的Cell，并将这个Cell返回。

        参数：
            - **key** (string) - 从CellDict中移除的Cell的键。

        异常：
            - **KeyError** - `key` 对应的Cell在CellDict中不存在。

    .. py:method:: update(cells)

        使用映射或者可迭代对象中的键值对来更新CellDict中已存在的Cell。

        参数：
            - **cells** (iterable) - 一个可迭代对象，通过它可迭代若干键值对(key, Cell)，其中键值对的类型为(string, Cell)；或者是一个从string到Cell的映射（字典）。Cell的类型不能为CellDict, CellList或者SequentialCell。key不能与类Cell中的属性重名，不能包含‘.’，不能是一个空串。

        .. note::
            如果 `cells` 是一个CellDict、一个OrderedDict或者是一个包含键值对的可迭代对象，那么新增元素的顺序在CellDict中仍会被保留。

        异常：
            - **TypeError** - 如果 `cells` 不是一个可迭代对象。
            - **TypeError** - 如果 `cells` 中的键值对不是可迭代对象。
            - **ValueError** - 如果 `cells` 中键值对的长度不是2。
            - **TypeError** - 如果 `cells` 中的cell是None。
            - **TypeError** - 如果 `cells` 中的cell的类型不是Cell。
            - **TypeError** - 如果 `cells` 中的cell的类型是CellDict，CellList或者SequentialCell。
            - **TypeError** - 如果 `cells` 中的key的类型不是String类型。
            - **KeyError** - 如果 `cells` 中的key与类Cell中的属性重名。
            - **KeyError** - 如果 `cells` 中的key包含“.”。
            - **KeyError** - 如果 `cells` 中的key是一个空串。

    .. py:method:: values()

        返回包含CellDict中所有Cell的可迭代对象。

        返回：
            一个可迭代对象。
