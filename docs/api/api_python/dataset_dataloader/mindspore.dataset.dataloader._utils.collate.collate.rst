mindspore.dataset.dataloader._utils.collate.collate
===================================================

.. py:function:: mindspore.dataset.dataloader._utils.collate.collate(batch, *, collate_fn_map=None)

    根据输入批数据元素的类型，从 `collate_fn_map` 所定义的类型到整理函数映射中，选择相应函数对批数据进行整理。

    批数据中的所有元素应该是相同类型。

    * 如果元素的类型在 `collate_fn_map` 中，或者元素是 `collate_fn_map` 中类型的子类，则使用相应函数进行数据整理；
    * 如果元素是映射（ :py:class:`~collections.abc.Mapping` ）类型，则按键分组整理：对每个键，收集批数据所有映射中该键
      对应的值，组成新的批数据，对其递归调用本函数，将结果作为该键新的值。批数据中各映射的键必须相同，各个键对应值的类型必须相同；
    * 如果元素是序列（ :py:class:`~collections.abc.Sequence` ）类型，则按位置分组整理：对每个位置，收集批数据所有序列中
      该位置对应的元素，组成新的批数据，对其递归调用本函数，将结果作为该位置新的元素。批数据中各序列的长度必须相同；
    * 否则将抛出异常，表明不支持该元素类型。

    每个整理函数需要一个 `batch` 位置参数和一个 `collate_fn_map` 关键字参数。

    参数：
        - **batch** (list) - 要整理的批数据。

    关键字参数：
        - **collate_fn_map** (Optional[dict[Union[type, tuple[type, ...]], Callable]]) - 从元素类型到相应整理函数的映射。
          默认值： ``None`` 。

    返回：
        :py:class:`~typing.Any` ，整理后的数据。
