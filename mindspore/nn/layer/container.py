# Copyright 2020 Huawei Technologies Co., Ltd
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
from collections import OrderedDict
from abc import abstractmethod
from ..cell import Cell

__all__ = ['SequentialCell', 'CellList']


def _valid_index(cell_num, index):
    if not isinstance(index, int):
        raise TypeError("Index {} is not int type")
    if not -cell_num <= index < cell_num:
        raise IndexError("Index should be a number in range [{}, {}), but got {}"
                         .format(-cell_num, cell_num, index))
    return index % cell_num


def _valid_cell(cell):
    if issubclass(cell.__class__, Cell):
        return True
    raise TypeError('Cell {} is not subclass of Cell'.format(cell))


class _CellListBase():
    """
    An interface for base the cell as list.

    The sequential cell may be iterated using the construct method using for-in statement.
    But there are some scenarios that the construct method built-in does not fit.
    For convenience, we provide an interface that indicates the sequential
    cell may be interpreted as list of cells, so it can be accessed using
    iterator or subscript when a sequential cell instantiate is accessed
    by iterator or subscript , it will be interpreted as a list of cells.
    """
    def __init__(self):
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
    Sequential cell container.

    A list of Cells will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of cells can also be passed in.

    Args:
        args (list, OrderedDict): List of subclass of Cell.

    Raises:
        TypeError: If the type of the argument is not list or OrderedDict.

    Inputs:
        - **input** (Tensor) - Tensor with shape according to the first Cell in the sequence.

    Outputs:
        Tensor, the output Tensor with shape depending on the input and defined sequence of Cells.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        >>> relu = nn.ReLU()
        >>> seq = nn.SequentialCell([conv, relu])
        >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        >>> output = seq(x)
        >>> print(output)
        [[[[27. 27.]
           [27. 27.]]
          [[27. 27.]
           [27. 27.]]]]
    """
    def __init__(self, *args):
        super(SequentialCell, self).__init__()
        if len(args) == 1:
            cells = args[0]
            if isinstance(cells, list):
                for index, cell in enumerate(cells):
                    self.insert_child_to_cell(str(index), cell)
            elif isinstance(cells, OrderedDict):
                for name, cell in cells.items():
                    self.insert_child_to_cell(name, cell)
            else:
                raise TypeError('Cells must be list or orderedDict')
        else:
            for index, cell in enumerate(args):
                self.insert_child_to_cell(str(index), cell)
        self.cell_list = list(self._cells.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(
                OrderedDict(list(self._cells.items())[index]))
        index = _valid_index(len(self), index)
        return list(self._cells.values())[index]

    def __setitem__(self, index, cell):
        if _valid_cell(cell):
            index = _valid_index(len(self), index)
            key = list(self._cells.keys())[index]
            self._cells[key] = cell
            self.cell_list = list(self._cells.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            key = list(self._cells.keys())[index]
            del self._cells[key]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.cell_list = list(self._cells.values())

    def __len__(self):
        return len(self._cells)

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def append(self, cell):
        """Appends a given cell to the end of the list.

        Examples:
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
        if _valid_cell(cell):
            self._cells[str(len(self))] = cell
        self.cell_list = list(self._cells.values())
        return self

    def construct(self, input_data):
        for cell in self.cell_list:
            input_data = cell(input_data)
        return input_data


class CellList(_CellListBase, Cell):
    """
    Holds Cells in a list.

    CellList can be used like a regular Python list, support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__iadd__',
    but cells it contains are properly registered, and will be visible by all Cell methods.

    Args:
        args (list, optional): List of subclass of Cell.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> conv = nn.Conv2d(100, 20, 3)
        >>> bn = nn.BatchNorm2d(20)
        >>> relu = nn.ReLU()
        >>> cell_ls = nn.CellList([bn])
        >>> cell_ls.insert(0, conv)
        >>> cell_ls.append(relu)
        >>> cell_ls
        CellList<
          (0): Conv2d<input_channels=100, ..., bias_init=None>
          (1): BatchNorm2d<num_features=20, ..., moving_variance=Parameter (name=variance)>
          (2): ReLU<>
          >
    """
    def __init__(self, *args):
        _CellListBase.__init__(self)
        Cell.__init__(self)
        if len(args) == 1:
            self.extend(args[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(list(self._cells.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            return self._cells[str(index)]
        raise TypeError('Index {} is not int type or slice type'.format(index))

    def __setitem__(self, index, cell):
        if not isinstance(index, int) and _valid_cell(cell):
            raise TypeError('Index {} is not int type'.format(index))
        index = _valid_index(len(self), index)
        self._cells[str(index)] = cell

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            del self._cells[str(index)]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        # adjust orderedDict
        temp_dict = OrderedDict()
        for idx, cell in enumerate(self._cells.values()):
            temp_dict[str(idx)] = cell
        self._cells = temp_dict

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells.values())

    def __iadd__(self, cells):
        self.extend(cells)
        return self

    def insert(self, index, cell):
        """Inserts a given cell before a given index in the list."""
        idx = _valid_index(len(self), index)
        _valid_cell(cell)
        length = len(self)
        while length > idx:
            self._cells[str(length)] = self._cells[str(length - 1)]
            length -= 1
        self._cells[str(idx)] = cell

    def extend(self, cells):
        """
        Appends cells from a Python iterable to the end of the list.

        Raises:
            TypeError: If the cells are not a list of subcells.
        """
        if not isinstance(cells, list):
            raise TypeError('Cells {} should be list of subcells'.format(cells))
        for cell in cells:
            if _valid_cell(cell):
                self._cells[str(len(self))] = cell
        return self

    def append(self, cell):
        """Appends a given cell to the end of the list."""
        if _valid_cell(cell):
            self._cells[str(len(self))] = cell
        return self

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for cell in self._cells.values():
            cell.set_grad(flag)

    def construct(self, *inputs):
        raise NotImplementedError
