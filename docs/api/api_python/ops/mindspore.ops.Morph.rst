mindspore.ops.Morph
==============================

.. py:class:: mindspore.ops.Morph(fn, infer_shape, infer_dtype, bprop_fn=None)

    `Morph` 算子用于对用户自定义函数 `fn` 进行封装，允许其被当做自定义算子使用。

    `Morph` 算子主要适用于静态图自定义图优化场景。例如：分布式自动并行场景可以使用Morph封装非规整集合通信（如 :func:`mindspore.ops.AlltoAllV`）。

    `Morph` 算子作用于输入时，实际上是其内封装的自定义函数 `fn` 作用于输入。

    `Morph` 算子与 :func:`mindspore.ops.Custom` 的主要区别在于，前者会在自动微分前被展开替换为用户自定义 `fn`，故无需实现反向函数。

    .. note::
        - 本算子只支持图模式。
        - `Morph` 允许用户使用自定义反向函数（通过参数 `bprop_fn`）。
        - `fn` 和 `bprop_fn` 必须满足图模式语法约束。
        - 用户自定义函数不支持 `vararg`、`kwarg`、`kwonlyargs` 和自由变量。

    参数：
        - **fn** (Function) - MindSpore Function，用户自定义函数。
        - **infer_shape** (Function) - MindSpore Function，用户自定义 `infer_shape` 函数。
        - **infer_dtype** (Function) - MindSpore Function，用户自定义 `infer_dtype` 函数。
        - **bprop_fn** (Function，可选) - MindSpore Function，用户自定义的反向函数。默认值: ``None``。

    输入：
        用户自定义 `fn` 的输入。

    输出：
        用户自定义 `fn` 的输出。

    异常：
        - **RuntimeError** - 如果算子在非图模式下被使用。
