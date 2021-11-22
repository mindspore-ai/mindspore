apply(apply_func)

    对数据集对象执行给定操作函数。

    参数：
        apply_func (function)：传入`Dataset`对象作为参数，并将返回处理后的`Dataset`对象。

    返回：
        执行了给定操作函数的数据集对象。

    示例：
        >>> # dataset是数据集类的实例化对象
        >>>
        >>> # 声明一个名为apply_func函数，其返回值是一个Dataset对象
        >>> def apply_func(data)：
        ...     data = data.batch(2)
        ...     return data
        >>>
        >>> # 通过apply操作调用apply_func函数
        >>> dataset = dataset.apply(apply_func)

    异常：
        TypeError：apply_func不是一个函数。
        TypeError：apply_func未返回Dataset对象。