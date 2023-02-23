mindspore.vmap
==================

.. py:function:: mindspore.vmap(fn, in_axes=0, out_axes=0)

    自动向量化（Vectorizing Map，vmap），是一种用于沿参数轴映射函数 `fn` 的高阶函数。

    Vmap由Jax率先提出，它消除了算子对batch维度的限制，并提供更加方便、统一的运算符表达。同时，用户还可以与 :func:`mindspore.grad` 等其它功能模块组合使用，提高开发效率。

    此外，由于自动向量化并不在函数外部执行循环，而是将循环逻辑下沉至函数的各个原语操作中，以获得更好的性能。当与图算融合特性相结合时，执行效率将进一步提高。

    .. warning::
        这是一个实验性算子，可能被更改或删除。

    .. note::
        - vmap的能力来自于原语的VmapRules实现。虽然我们为用户自定义算子设计了一个通用规则，但是我们无法保证它对所有算子都能很好地工作，用户需知晓使用风险。如果想要获得更好的性能，请提前花一些时间参阅教程为自定义算子实现特定的VmapRule。
        - 当在vmap作用域内调用随机数生成方法时，每次在向量函数之间生成相同的随机数。如果希望每个向量分支使用不同的随机数，需要提前从外部生成一批随机数，然后将其传入vmap。

    参数：
        - **fn** (Union[Cell, Function, CellList]) - 待沿参数轴映射的函数，该函数至少拥有一个输入参数并且返回值为一个或多个Tensor或Tensor支持的数据类型。当 `fn` 的类型是CellList时，为模型集成场景，需要确保每个单元的结构相同，并且单元数量与映射轴索引对应的size（ `axis_size` ）一致。
        - **in_axes** (Union[int, list, tuple]) - 指定输入参数映射的轴索引。如果 `in_axes` 是一个整数，则 `fn` 的所有输入参数都将根据此轴索引进行映射。
          如果 `in_axes` 是一个tuple或list，仅支持由整数或None组成，则其长度应与 `fn` 的输入参数的个数一致，分别表示相应位置参数的映射轴索引。
          请注意，每个参数对应的整数轴索引的取值范围必须在 :math:`[-ndim, ndim)` 中，其中 `ndim` 是参数的维度。None表示不沿任何轴映射。并且 `in_axes` 中必须至少有一个位置参数的映射轴索引不为None。所有参数的映射轴索引对应的size（ `axis_size` ）必须相等。默认值：0。
        - **out_axes** (Union[int, list, tuple]) - 指定映射轴呈现在输出中的索引位置。如果 `out_axes` 是一个整数，则 `fn` 的所有输出都根据此axis指定。
          如果 `out_axes` 是一个tuple或list，仅支持由整数或None组成，其长度应与 `fn` 的输出个数相等。
          请注意，每个输出对应的整数轴索引的取值范围必须在 :math:`[-ndim, ndim)` 中，其中 `ndim` 是 `vmap` 映射后的函数的输出的维度。
          所有具有非None映射轴的输出只能指定非None的 `out_axes` ，如果具有None映射轴的输出指定非None的 `out_axes` ，结果将沿映射轴进行广播。默认值：0。

    返回：
        Function，返回 `fn` 的自动向量化后的函数。此函数的输入参数和输出与 `fn` 的相对应，但它在 `in_axes` 和 `out_axes` 指定的位置新增了额外的批处理维度。

    异常：
        - **RuntimeError** -

          - 如果 `in_axes` 或 `out_axes` 中的基本元素不是None或整数。
          - 如果 `in_axes` 或 `out_axes` 中的所有基本元素均为None。
          - 如果 `in_axes` 不是单个整数，并且 `in_axes` 的长度不等于输入参数的个数。
          - 如果 `out_axes` 不是单个整数，并且 `out_axes` 的长度不等于输出个数。
          - 如果 `vmap` 范围内每个参数的 `axis_size` 不相等。
          - 如果 `in_axes` 或 `out_axes` 中的 `axis` 超出边界限制。
