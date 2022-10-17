import sys
import numpy as np

import mindspore.context as context
from mindspore import Tensor
from mindspore import ms_function


@ms_function
def func(input_x, input_y):
    output = input_x + input_x * input_y
    return output


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, enable_compile_cache=True, compile_cache_path=sys.argv[1])
    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    res = func(x, y)
    print("{", res, "}")
    print("{", res.asnumpy().shape, "}")
    context.set_context(enable_compile_cache=False)
