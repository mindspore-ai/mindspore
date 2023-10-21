# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""container"""
from __future__ import absolute_import

from collections import OrderedDict, abc
from abc import abstractmethod

from mindspore.nn.cell import Cell

__all__ = ['SequentialCell', 'CellList', 'CellDict']


def _valid_index(cell_num, index, op_name=None):
    """Internal function, used to detect the value and type of index."""
    msg_prefix = f"For '{op_name}', the" if op_name else "The"
    if not isinstance(index, int):
        raise TypeError(f"{msg_prefix} type of 'index' must be int, but got {type(index).__name__}.")
    if not -cell_num <= index < cell_num:
        raise IndexError(f"{msg_prefix} value of 'index' must be a number in range [{-cell_num}, {cell_num}), "
                         f"but got {index}.")
    return index % cell_num


def _valid_index_for_inserting(cell_num, index, op_name=None):
    """
    Internal function, used to detect the value and type of index for inserting Cell in
    SequentialCell or CellList.
    """
    msg_prefix = f"For '{op_name}', the" if op_name else "The"
    if not isinstance(index, int):
        raise TypeError(f"{msg_prefix} type of 'index' must be int, but got {type(index).__name__}.")
    if not -cell_num <= index <= cell_num:
        raise IndexError(f"{msg_prefix} value of 'index' must be a number in range [{-cell_num}, {cell_num}], "
                         f"but got {index}.")
    return index % cell_num if (cell_num != 0 and index != cell_num) else index


def _valid_cell(cell, op_name=None):
    """Internal function, used to check whether the input cell is a subclass of Cell."""
    if issubclass(cell.__class__, Cell):
        return True
    msg_prefix = f"For '{op_name}'," if op_name else ""
    raise TypeError(f'{msg_prefix} each cell must be subclass of Cell, but got {type(cell).__name__}.')


def _get_prefix_and_index(cells):
    """get prefix and index of parameter name in sequential cell or cell list."""
    prefix = ""
    index = 0
    if not cells:
        return prefix, index

    cell_list = list(cells.items())
    first_param, first_key = None, None
    second_param, second_key = None, None
    for key, cell in cell_list:
        try:
            _, param = next(cell.parameters_and_names())
        except StopIteration:
            continue
        if first_param is None:
            first_param = param
            first_key = key
            continue
        second_param = param
        second_key = key
        break

    if first_param is None:
        return prefix, index

    split_names = first_param.name.split(".")
    for idx, name in enumerate(split_names):
        if name == first_key:
            prefix = ".".join(split_names[:idx])
            prefix = prefix + "." if prefix else prefix
            index = idx
            if second_param is not None and second_param.name.split(".")[idx] == second_key:
                break
    return prefix, index


class _CellListBase:
    """
    An interface for base the Cell as list.

    The sequential Cell may be iterated using the construct method using for-in statement.
    But there are some scenarios that the construct method built-in does not fit.
    For convenience, we provide an interface that indicates the sequential
    Cell may be interpreted as list of Cells, so it can be accessed using
    iterator or subscript when a sequential Cell instantiate is accessed
    by iterator or subscript, it will be interpreted as a list of Cells.
    """
    def __init__(self):
        """Initialize _CellListBase."""
        self.__cell_as_list__ = True

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def construct(self):
        raise NotImplementedError


class SequentialCell(Cell):
    """
    Sequential Cell container. For more details about Cell, please refer to
    `Cell <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_.

    A list of Cells will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of cells can also be passed in.

    Args:
        args (list, OrderedDict): List or OrderedDict of subclass of Cell.

    Inputs:
        - **x** (Tensor) - Tensor with shape according to the first Cell in the sequence.

    Outputs:
        Tensor, the output Tensor with shape depending on the input `x` and defined sequence of Cells.

    Raises:
        TypeError: If the type of the `args` is not list or OrderedDict.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>>
        >>> conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        >>> relu = nn.ReLU()
        >>> seq = nn.SequentialCell([conv, relu])
        >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype = mindspore.float32)
        >>> output = seq(x)
        >>> print(output)
        [[[[27. 27.]
           [27. 27.]]
          [[27. 27.]
           [27. 27.]]]]
        >>> from collections import OrderedDict
        >>> d = OrderedDict()
        >>> d["conv"] = conv
        >>> d["relu"] = relu
        >>> seq = nn.SequentialCell(d)
        >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        >>> output = seq(x)
        >>> print(output)
        [[[[27. 27.]
           [27. 27.]]
          [[27. 27.]
           [27. 27.]]]]
    """
    def __init__(self, *args):
        """Initialize SequentialCell."""
        super(SequentialCell, self).__init__()
        self._is_dynamic_name = []
        if len(args) == 1:
            cells = args[0]
            if isinstance(cells, Cell):
                cell = cells
                self.insert_child_to_cell(str(0), cell)
                cell.update_parameters_name(str(0) + ".")
                self._is_dynamic_name.append(True)
            elif isinstance(cells, list):
                for index, cell in enumerate(cells):
                    self.insert_child_to_cell(str(index), cell)
                    cell.update_parameters_name(str(index) + ".")
                    self._is_dynamic_name.append(True)
            elif isinstance(cells, OrderedDict):
                for name, cell in cells.items():
                    self.insert_child_to_cell(name, cell)
                    cell.update_parameters_name(name + ".")
                    self._is_dynamic_name.append(False)
            else:
                raise TypeError(f"For '{self.__class__.__name__}', the 'args[0]' must be Cell, list or orderedDict, "
                                f"but got {type(cells).__name__}")
        else:
            for index, cell in enumerate(args):
                self.insert_child_to_cell(str(index), cell)
                cell.update_parameters_name(str(index) + ".")
                self._is_dynamic_name.append(True)
        self.cell_list = list(self._cells.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(
                OrderedDict(list(self._cells.items())[index]))
        index = _valid_index(len(self), index, self.__class__.__name__)
        return list(self._cells.values())[index]

    def __setitem__(self, index, cell):
        cls_name = self.__class__.__name__
        if _valid_cell(cell, cls_name):
            prefix, _ = _get_prefix_and_index(self._cells)
            index = _valid_index(len(self), index, cls_name)
            key = list(self._cells.keys())[index]
            self._cells[key] = cell
            cell.update_parameters_name(prefix + key + ".")
            self.cell_list = list(self._cells.values())

    def __delitem__(self, index):
        cls_name = self.__class__.__name__
        if isinstance(index, int):
            index = _valid_index(len(self), index, cls_name)
            key = list(self._cells.keys())[index]
            del self._cells[key]
            del self._is_dynamic_name[index]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
            del self._is_dynamic_name[index]
        else:
            raise TypeError(f"For '{cls_name}', the type of index must be int type or slice type, "
                            f"but got {type(index).__name__}")
        prefix, key_index = _get_prefix_and_index(self._cells)
        temp_dict = OrderedDict()
        for idx, key in enumerate(self._cells.keys()):
            cell = self._cells[key]
            if self._is_dynamic_name[idx]:
                for _, param in cell.parameters_and_names():
                    param.name = prefix + str(idx) + "." + ".".join(param.name.split(".")[key_index+1:])
                temp_dict[str(idx)] = cell
            else:
                temp_dict[key] = cell
        self._cells = temp_dict
        self.cell_list = list(self._cells.values())

    def __bool__(self):
        return len(self._cells) != 0

    def __len__(self):
        return len(self._cells)

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def append(self, cell):
        """
        Appends a given Cell to the end of the list.

        Args:
            cell(Cell): The Cell to be appended.

        Examples:
            >>> import mindspore
            >>> from mindspore import Tensor, nn
            >>> import numpy as np
            >>>
            >>> conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
            >>> bn = nn.BatchNorm2d(2)
            >>> relu = nn.ReLU()
            >>> seq = nn.SequentialCell([conv, bn])
            >>> seq.append(relu)
            >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
            >>> output = seq(x)
            >>> print(output)
            [[[[26.999863 26.999863]
               [26.999863 26.999863]]
              [[26.999863 26.999863]
               [26.999863 26.999863]]]]
        """
        if _valid_cell(cell, self.__class__.__name__):
            prefix, _ = _get_prefix_and_index(self._cells)
            cell.update_parameters_name(prefix + str(len(self)) + ".")
            self._is_dynamic_name.append(True)
            self._cells[str(len(self))] = cell
        self.cell_list = list(self._cells.values())

    def construct(self, input_data):
        for cell in self.cell_list:
            input_data = cell(input_data)
        return input_data

    def _insert(self, index, cell):
        """
        Inserts a given Cell before a given index in the list.

        Args:
            index(int): The Insert index in the CellList.
            cell(Cell): The Cell to be inserted.
        """
        cls_name = self.__class__.__name__
        idx = _valid_index_for_inserting(len(self), index, cls_name)
        _valid_cell(cell, cls_name)
        length = len(self)
        prefix, key_index = _get_prefix_and_index(self._cells)
        while length > idx:
            if self._auto_prefix:
                tmp_cell = self._cells[str(length-1)]
                for _, param in tmp_cell.parameters_and_names():
                    param.name = f'{prefix}{str(length)}{"."}{".".join(param.name.split(".")[key_index+1:])}'
            self._cells[str(length)] = self._cells[str(length - 1)]
            length -= 1
        self._cells[str(idx)] = cell
        if self._auto_prefix:
            cell.update_parameters_name(prefix + str(idx) + ".")
        self.cell_list = list(self._cells.values())
        self._is_dynamic_name.insert(index, True)


class CellList(_CellListBase, Cell):
    """
    Holds Cells in a list. For more details about Cell, please refer to
    `Cell <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_.

    CellList can be used like a regular Python list, the Cells it contains have been initialized and
    the types of Cells it contains can not be CellDict.
    Unlike the SequentialCell, the cells in CellList are not connected.

    Args:
        args (list, optional): List of subclass of Cell.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>>
        >>> conv = ms.nn.Conv2d(100, 20, 3)
        >>> bn = ms.nn.BatchNorm2d(20)
        >>> relu = ms.nn.ReLU()
        >>> cell_ls = ms.nn.CellList([bn])
        >>> cell_ls.insert(0, conv)
        >>> cell_ls.append(relu)
        >>> cell_ls.extend([relu, relu])
        >>> cell_ls_3 = cell_ls[3]
        >>> input1 = ms.Tensor(np.ones([2, 3]), ms.float32)
        >>> output = cell_ls_3(input1)
        >>> print(output)
        [[1. 1. 1.]
        [1. 1. 1.]]
    """
    def __init__(self, *args, **kwargs):
        """Initialize CellList."""
        auto_prefix = kwargs["auto_prefix"] if "auto_prefix" in kwargs.keys() else True
        _CellListBase.__init__(self)
        Cell.__init__(self, auto_prefix)
        if len(args) == 1:
            self.extend(args[0])

    def __getitem__(self, index):
        cls_name = self.__class__.__name__
        if isinstance(index, slice):
            return self.__class__(list(self._cells.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index, cls_name)
            return self._cells[str(index)]
        raise TypeError(f"For '{cls_name}', the type of 'index' must be int or slice, "
                        f"but got {type(index).__name__}.")

    def __setitem__(self, index, cell):
        cls_name = self.__class__.__name__
        if not isinstance(index, int) and _valid_cell(cell, cls_name):
            raise TypeError(f"For '{cls_name}', the type of 'index' must be int, "
                            f"but got {type(index).__name__}.")
        index = _valid_index(len(self), index, cls_name)
        if self._auto_prefix:
            prefix, _ = _get_prefix_and_index(self._cells)
            cell.update_parameters_name(prefix + str(index) + ".")
        self._cells[str(index)] = cell

    def __delitem__(self, index):
        cls_name = self.__class__.__name__
        if isinstance(index, int):
            index = _valid_index(len(self), index, cls_name)
            del self._cells[str(index)]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
        else:
            raise TypeError(f"For '{cls_name}', the type of 'index' must be int or slice, "
                            f"but got {type(index).__name__}.")
        # adjust orderedDict
        prefix, key_index = _get_prefix_and_index(self._cells)
        temp_dict = OrderedDict()
        for idx, cell in enumerate(self._cells.values()):
            if self._auto_prefix:
                for _, param in cell.parameters_and_names():
                    param.name = prefix + str(idx) + "." + ".".join(param.name.split(".")[key_index+1:])
            temp_dict[str(idx)] = cell
        self._cells = temp_dict

    def __bool__(self):
        return len(self._cells) != 0

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells.values())

    def __iadd__(self, cells):
        self.extend(cells)
        return self

    def insert(self, index, cell):
        """
        Inserts a given Cell before a given index in the list.

        Args:
            index(int): The Insert index in the CellList.
            cell(Cell): The Cell to be inserted.
        """
        cls_name = self.__class__.__name__
        idx = _valid_index_for_inserting(len(self), index, cls_name)
        _valid_cell(cell, cls_name)
        length = len(self)
        prefix, key_index = _get_prefix_and_index(self._cells)
        while length > idx:
            if self._auto_prefix:
                tmp_cell = self._cells[str(length-1)]
                for _, param in tmp_cell.parameters_and_names():
                    param.name = prefix + str(length) + "." + ".".join(param.name.split(".")[key_index+1:])
            self._cells[str(length)] = self._cells[str(length - 1)]
            length -= 1
        self._cells[str(idx)] = cell
        if self._auto_prefix:
            cell.update_parameters_name(prefix + str(idx) + ".")

    def extend(self, cells):
        """
        Appends Cells from a Python iterable to the end of the list.

        Args:
            cells(list): The Cells to be extended, the types of Cells can not be CellDict.

        Raises:
            TypeError: If the argument cells are not a list of Cells.
        """
        cls_name = self.__class__.__name__
        if not isinstance(cells, list):
            raise TypeError(f"For '{cls_name}', the new cells wanted to append "
                            f"should be instance of list, but got {type(cells).__name__}.")
        prefix, _ = _get_prefix_and_index(self._cells)
        for cell in cells:
            if isinstance(cell, CellDict):
                raise TypeError(f"For '{cls_name}', the type of cell can not be CellDict, "
                                f"but got {type(cell).__name__}.")
            if _valid_cell(cell, cls_name):
                if self._auto_prefix:
                    cell.update_parameters_name(prefix + str(len(self)) + ".")
                self._cells[str(len(self))] = cell
        return self

    def append(self, cell):
        """
        Appends a given Cell to the end of the list.

        Args:
            cell(Cell): The subcell to be appended.
        """
        if _valid_cell(cell, self.__class__.__name__):
            if self._auto_prefix:
                prefix, _ = _get_prefix_and_index(self._cells)
                cell.update_parameters_name(prefix + str(len(self)) + ".")
            self._cells[str(len(self))] = cell

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def construct(self, *inputs):
        raise NotImplementedError


class _CellDictBase:
    """
    An interface for base the Cell as dict.

    The sequential Cell may be iterated using the construct method using for-in statement.
    But there are some scenarios that the construct method built-in does not fit.
    For convenience, we provide an interface that indicates the sequential
    Cell may be interpreted as dict of Cells, so it can be accessed using
    key when a sequential Cell instantiate is accessed by key,
    it will be interpreted as a dict of Cells.
    """
    def __init__(self):
        """Initialize _CellDictBase."""
        self.__cell_as_dict__ = True

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def construct(self):
        raise NotImplementedError


class CellDict(_CellDictBase, Cell):
    """
    Holds Cells in a dictionary. For more details about `Cell` , please refer to :class:`mindspore.nn.Cell` .

    `CellDict` can be used like a regular Python dictionary.

    Args:
        args (iterable, optional): An iterable of key-value pairs of (key, Cell), the type of key-value pairs is
                                   (string, Cell); Or a mapping(dictionary) from string to Cell.
                                   The type of Cell can not be CellDict, CellList or SequentialCell.
                                   The key can not be same with the attributes of class Cell, can not contain '.',
                                   can not be an empty string.
                                   The key of type string is used to search corresponding Cell in the CellDict.
        kwargs (dict): Reserved for keyword argument to be expanded.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import collections
        >>> from collections import OrderedDict
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>>
        >>> cell_dict = nn.CellDict({'conv': nn.Conv2d(10, 6, 5),
        ...                          'relu': nn.ReLU(),
        ...                          'max_pool2d': nn.MaxPool2d(kernel_size=4, stride=4)})
        >>> print(len(cell_dict))
        3
        >>> cell_dict.clear()
        >>> print(len(cell_dict))
        0
        >>> ordered_cells = OrderedDict([('conv', nn.Conv2d(10, 6, 5, pad_mode='valid')),
        ...                              ('relu', nn.ReLU()),
        ...                              ('max_pool2d', nn.MaxPool2d(kernel_size=2, stride=2))])
        >>> cell_dict.update(ordered_cells)
        >>> x = Tensor(np.ones([1, 10, 6, 10]), ms.float32)
        >>> for cell in cell_dict.values():
        ...     x = cell(x)
        >>> print(x.shape)
        (1, 6, 1, 3)
        >>> x = Tensor(np.ones([1, 10, 6, 10]), ms.float32)
        >>> for item in cell_dict.items():
        ...     x = item[1](x)
        >>> print(x.shape)
        (1, 6, 1, 3)
        >>> print(cell_dict.keys())
        odict_keys(['conv', 'relu', 'max_pool2d'])
        >>> pop_cell = cell_dict.pop('conv')
        >>> x = Tensor(np.ones([1, 10, 6, 5]), ms.float32)
        >>> x = pop_cell(x)
        >>> print(x.shape)
        (1, 6, 2, 1)
        >>> print(len(cell_dict))
        2
    """
    def __init__(self, *args, **kwargs):
        """Initialize CellDict."""
        auto_prefix = kwargs["auto_preifx"] if "auto_prefix" in kwargs.keys() else True
        _CellDictBase.__init__(self)
        Cell.__init__(self, auto_prefix)
        if len(args) == 1:
            self.update(args[0])

    def __getitem__(self, key):
        return self._cells[key]

    def __setitem__(self, key, cell):
        self._validate_key(key)
        self._validate_cell_type(cell)
        self._update_cell_para_name(key, cell)
        self._cells[key] = cell

    def __delitem__(self, key):
        del self._cells[key]

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells)

    def __contains__(self, key):
        return key in self._cells

    def _validate_key(self, key):
        """validate key."""
        cls_name = self.__class__.__name__
        if not isinstance(key, str):
            raise TypeError(f"For '{cls_name}', the type of key should be string "
                            f"but got {type(key).__name__}.")
        if hasattr(self, key) and key not in self._cells:
            raise KeyError(f"For '{cls_name}', the key can not be same with the attributes of Cell, "
                           f"but got key {key}.")
        if '.' in key:
            raise KeyError(f"For '{cls_name}', key can not contain \".\", "
                           f"but got key {key}")
        if key == '':
            raise KeyError(f"For '{cls_name}', key can not be empty string \"\", "
                           f"but got key {key}")

    def _validate_cell_type(self, cell):
        """validate cell type."""
        cls_name = self.__class__.__name__
        if cell is None:
            raise TypeError(f"For '{cls_name}', cell can not be None.")
        if not isinstance(cell, Cell):
            raise TypeError(f"For '{cls_name}', the type of cell should be Cell, "
                            f"but got {type(cell).__name__}.")
        if isinstance(cell, (CellDict, CellList, SequentialCell)):
            raise TypeError(f"For '{cls_name}', the type of cell can not be CellDict, CellList or SequentialCell, "
                            f"but got {type(cell).__name__}.")

    def _update_cell_para_name(self, key, cell):
        """update cell para name."""
        if self._auto_prefix:
            prefix, _ = _get_prefix_and_index(self._cells)
            cell.update_parameters_name(prefix + key + ".")

    def clear(self):
        """
        Remove all Cells from the CellDict.
        """
        return self._cells.clear()

    def pop(self, key):
        """
        Remove key from the CellDict and return its cell.

        Args:
            key (string): key to pop from the CellDict.

        Raises:
            KeyError: If `key` not exist in CellDict when attempt to access cell.
        """
        value = self[key]
        del self[key]
        return value

    def keys(self):
        """
        Return an iterable of the CellDict keys.

        Returns:
            An iterable object.
        """
        return self._cells.keys()

    def values(self):
        """
        Return an iterable of the CellDict values.

        Returns:
            An iterable object.
        """
        return self._cells.values()

    def items(self):
        """
        Return an iterable of the CellDict key-value pairs.

        Returns:
            An iterable object.
        """
        return self._cells.items()

    def update(self, cells):
        """
        Update the CellDict by overwriting the existing keys with the key-value pairs from a mapping or an iterable.

        Args:
            cells (iterable): An iterable of key-value pairs of (key, Cell), the type of key-value pairs is
                              (string, Cell); Or a mapping(dictionary) from string to Cell.
                              The type of Cell can not be CellDict, CellList or SequentialCell.
                              The key can not be same with the attributes of class Cell, can not contain '.',
                              can not be an empty string.

        Note:
            If the `cells` is a CellDict, an OrderedDict or an iterable containing key-value pairs,
            the order of newly added elements is maintained.

        Raises:
            TypeError: If `cells` is not an iterable object.
            TypeError: If key-value pairs in `cells` are not iterable objects.
            ValueError: If the length of key-value pairs in `cells` is not 2.
            TypeError: If the cell in `cells` is None.
            TypeError: If the type of cell in `cells` is not Cell.
            TypeError: If the type of cell in `cells` is CellDict, CellList or SequentialCell.
            TypeError: If the type of key in `cells` is not string.
            KeyError: If the key in `cells` is same with the attributes of class Cell.
            KeyError: If the key in `cells` contain ".".
            KeyError: If the key in `cells` is an empty string.
        """
        if not isinstance(cells, abc.Iterable):
            raise TypeError("CellDict.update() should be called with an "
                            "iterable of key-value pairs, but got " +
                            type(cells).__name__)
        if isinstance(cells, (OrderedDict, CellDict, abc.Mapping)):
            for key, cell in cells.items():
                self[key] = cell
        else:
            for id, k_v in enumerate(cells):
                if not isinstance(k_v, abc.Iterable):
                    raise TypeError("CellDict update sequence element "
                                    "#" + str(id) + " should be Iterable; but got " +
                                    type(k_v).__name__)
                if len(k_v) != 2:
                    raise ValueError("CellDict update sequence element "
                                     "#" + str(id) + ", length should be 2; but has length " +
                                     str(len(k_v)))
                self[k_v[0]] = k_v[1]

    def construct(self, *inputs):
        raise NotImplementedError
