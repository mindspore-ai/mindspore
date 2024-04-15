# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Symbol implementation."""

__all__ = ['Symbol']


class Symbol:
    r"""
    Symbol is a data structure to indicate the symbolic info of shape.

    For dynamic shape networks, compared with only setting the unknown dimensions ( ``None`` ) in `Tensor` , providing
    more symbolic shape info can help the framework better optimize the computation graph, to improve the performance of
    network execution.

    Args:
        max (int): The maximum length of this dimension, which is valid when it's greater than `min`. Default: ``0`` .
        min (int): The minimum length of this dimension. Default: ``1`` .
        divisor (int): The divisor( :math:`d` ). When `remainder` is 0, it means this dimension can be divided by
            :math:`d` . Default: ``1`` .
        remainder (int): The remainder( :math:`r` ) when symbol is represented by :math:`d * N + r, N \ge 1` .
            Default: ``0`` .
        unique (bool): When the symbol object is used multiple times, if `unique` is ``True`` , the shape items of this
            symbol are considered to be same length, otherwise only symbol info is shared by multiple dimensions.
            Default: ``False`` .

    Outputs:
        Symbol.

    Raises:
        TypeError: If `max`, `min`, `divisor`, `remainder` is not an int.
        TypeError: If `unique` is not a bool.
        ValueError: If `min` is not positive value.
        ValueError: If `divisor` is not positive value.
        ValueError: If `remainder` is not in the range :math:`[0, d)` .

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor, Symbol
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.abs = ms.ops.Abs()
        ...     def construct(self, x):
        ...         return self.abs(x)
        ...
        >>> net = Net()
        >>> s1 = Symbol(divisor=8, remainder=1)
        >>> s2 = Symbol(max=32, unique=True)
        >>> dyn_t = Tensor(shape=(None, s1, s1, s2, s2), dtype=ms.float32)
        >>> net.set_inputs(dyn_t)
        >>> # the shape values of last two dimensions must be equal, because "s2" is set to "unique"
        >>> net(Tensor(np.random.randn(1, 9, 17, 32, 32), dtype=ms.float32)).shape
        (1, 9, 17, 32, 32)
        >>> net(Tensor(np.random.randn(8, 25, 9, 30, 30), dtype=ms.float32)).shape
        (8, 25, 9, 30, 30)
    """

    def __init__(self, max=0, min=1, divisor=1, remainder=0, unique=False, **kawgs):
        Symbol._check_args_type(max, min, divisor, remainder, unique)
        if min <= 0:
            raise ValueError("For 'Symbol', the 'min' value should be positive, but got {}".format(min))
        if divisor <= 0:
            raise ValueError("For 'Symbol', the 'divisor' value should be positive, but got {}".format(divisor))
        if remainder < 0 or remainder >= divisor:
            raise ValueError(
                "For 'Symbol', the 'remainder' value should be in the range '[0, {})', but got {}".format(
                    divisor, remainder))
        self.max = max
        self.min = min
        self.divisor = divisor
        self.remainder = remainder
        self.unique = unique
        self.id = id(self)

    def __str__(self):
        return str(self.to_dict())

    @staticmethod
    def _check_args_type(maxv, minv, divisor, remainder, unique):
        """Check the type of arguments."""
        if not isinstance(maxv, int):
            raise TypeError(f"For 'Symbol', the argument 'max' must be int, but got {type(maxv)}")
        if not isinstance(minv, int):
            raise TypeError(f"For 'Symbol', the argument 'min' must be int, but got {type(minv)}")
        if not isinstance(divisor, int):
            raise TypeError(f"For 'Symbol', the argument 'divisor' must be int, but got {type(divisor)}")
        if not isinstance(remainder, int):
            raise TypeError(f"For 'Symbol', the argument 'remainder' must be int, but got {type(remainder)}")
        if not isinstance(unique, bool):
            raise TypeError(f"For 'Symbol', the argument 'unique' must be bool, but got {type(unique)}")

    # pylint: disable=missing-docstring
    def to_dict(self):
        # Convert the symbolic info to dictionary.
        # This method is not necessary to show in public api document, use comment instead of docstring.
        res = {}
        if self.max > self.min:
            res["max"] = self.max
        if self.min > self.divisor + self.remainder:  # the symbol is "d * N + r" and N >= 1
            res["min"] = self.min
        if self.divisor != 1:
            res["divisor"] = self.divisor
        if self.remainder != 0:
            res["remainder"] = self.remainder
        if self.unique:
            res["id"] = self.id
        return res
