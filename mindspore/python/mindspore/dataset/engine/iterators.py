# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Built-in iterators"""
from abc import abstractmethod
from copy import deepcopy
import collections.abc
import json
import os
import signal
import weakref
import numpy as np

import mindspore._c_dataengine as cde
from mindspore.common.tensor import Tensor
import mindspore.dataset.engine.offload as offload
from mindspore.dataset.core.config import get_debug_mode

from mindspore import log as logger

_ITERATOR_CLEANUP = False


def _set_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = True


def _unset_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = False


def check_iterator_cleanup():
    global _ITERATOR_CLEANUP
    return _ITERATOR_CLEANUP


ITERATORS_LIST = list()


def _cleanup():
    """Release all the Iterator."""
    _set_iterator_cleanup()
    for itr_ref in reversed(ITERATORS_LIST):
        itr = itr_ref()
        if itr is not None:
            itr.release()


class Iterator:
    """
    General Iterator over a dataset.

    Attributes:
        dataset: Dataset to be iterated over
    """

    def __init__(self, dataset, num_epochs=-1, output_numpy=False, do_copy=True):
        self._col_names = None

        # create a copy of tree and work on it.
        self.__ori_dataset = dataset

        self.ir_tree, self.dataset = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        if get_debug_mode():
            consumer = cde.PythonPullBasedIteratorConsumer(num_epochs)
        else:
            consumer = cde.PythonIteratorConsumer(num_epochs)
        consumer.Init(self.ir_tree)
        self._runtime_context.AssignConsumer(consumer)
        self._iterator = self._runtime_context.GetConsumer()
        self._output_numpy = output_numpy
        self._do_copy = do_copy

        self.__index = 0

        self.offload_model = None
        json_offload = json.loads(consumer.GetOffload())

        # See if GetOffload identified any operations set to be offloaded.
        if json_offload is not None:
            offload.check_concat_zip_dataset(self.__ori_dataset)
            self.offload_model = offload.GetOffloadModel(consumer, self.__ori_dataset.get_col_names())

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def __iter__(self):
        return self

    def stop(self):
        """
        Manually terminate Python iterator instead of relying on out of scope destruction.
        """
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_iterator') and self._iterator:
                self._runtime_context.Terminate()
                del self._iterator
            del self._runtime_context
            del self.dataset

            # get weakref which is dead
            dead_iterator = []
            for index, item in enumerate(ITERATORS_LIST):
                # item() == None indicate the object is dead
                # id(item()) == id(self) indicate del self
                if item() is None or id(item()) == id(self):
                    dead_iterator.append(index)

            # del dead weakref
            for index in reversed(dead_iterator):
                ITERATORS_LIST.pop(index)

    def release(self):
        self.stop()

    def __del__(self):
        self.release()

    @abstractmethod
    def _get_next(self):
        raise RuntimeError("Calling base class Iterator's get_next is invalid.")

    def __next__(self):
        if not self._runtime_context:
            logger.warning("Iterator does not have a running C++ pipeline." +
                           "It might because Iterator stop() had been called, or C++ pipeline crashed silently.")
            raise RuntimeError("Iterator does not have a running C++ pipeline.")

        # Note offload is applied inside _get_next() if applicable since get_next converts to output format
        data = self._get_next()
        if not data:
            if self.__index == 0:
                logger.warning("No records available.")
            if self.__ori_dataset.dataset_size is None:
                self.__ori_dataset.dataset_size = self.__index
            raise StopIteration
        self.__index += 1

        return data

    def __deepcopy__(self, memo):
        return self

    def _getters(self):
        """
        Get pipeline information.
        """
        getter = cde.TreeGetters()
        getter.Init(self.ir_tree)
        self._runtime_context.AssignConsumer(getter)
        self._col_names = getter.GetColumnNames()

    def get_col_names(self):
        """
        Get names of the columns in the dataset
        """
        if self._col_names is None:
            self._col_names = self.__ori_dataset.get_col_names()
        return self._col_names

    def _reset(self, step, epoch):
        """
        Reset the iterator to the given step number and epoch number.

        Args:
            step (int): Global step number
            epoch (int): Global epoch number
        """
        self._iterator.Reset(step, epoch)

    def __convert_python_to_tensor(self, obj):
        """
        Attempts to recursively convert a python object to tensor(s).

        Args:
            obj (any): the python object to be converted
        """
        if isinstance(obj, (np.ndarray, int, float, bool, str)):
            if self._do_copy:
                return Tensor(np.asarray(obj))
            return Tensor.from_numpy(np.asarray(obj))
        if isinstance(obj, dict):
            return {key: self.__convert_python_to_tensor(val) for key, val in obj.items()}
        if isinstance(obj, collections.abc.Iterable):
            return [self.__convert_python_to_tensor(item) for item in obj]
        # if we can't convert it to Tensor, return the object as is
        if self._do_copy:
            return deepcopy(obj)
        return obj

    def _transform_md_to_output(self, t):
        if self._output_numpy:
            if t.type().is_python():
                return t.as_python()
            return t.as_array()
        return self._transform_md_to_tensor(t)

    def _transform_md_to_tensor(self, t):
        if t.type().is_python():
            return self.__convert_python_to_tensor(t.as_python())
        array = t.as_array()
        if self._do_copy:
            return Tensor(array)
        return Tensor.from_numpy(array)

    def _transform_tensor_to_output(self, t):
        if self._output_numpy:
            return t.asnumpy()
        return t


class DictIterator(Iterator):
    """
    The derived class of Iterator with dict type.
    """

    def _get_next(self):
        """
        Returns the next record in the dataset as dictionary

        Returns:
            Dict, the next record in the dataset.
        """
        try:
            if self.offload_model is None:
                return {k: self._transform_md_to_output(t) for k, t in self._iterator.GetNextAsMap().items()}
            data = [self._transform_md_to_tensor(t) for t in self._iterator.GetNextAsList()]
            if data:
                data = offload.apply_offload_iterators(data, self.offload_model)
                # Create output dictionary after offload
                out_data = {}
                for i, col in enumerate(self.get_col_names()):
                    out_data[col] = self._transform_tensor_to_output(data[i])
                data = out_data
            return data

        except RuntimeError as err:
            # maybe "Out of memory" / "MemoryError" error
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.critical("Memory error occurred, process will exit.")
                os.kill(os.getpid(), signal.SIGKILL)
            raise err


class TupleIterator(Iterator):
    """
    The derived class of Iterator with list type.
    """

    def __init__(self, dataset, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        if columns is not None:
            if not isinstance(columns, list):
                columns = [columns]
            dataset = dataset.project(columns)
        super().__init__(dataset, num_epochs, output_numpy, do_copy)

    def _get_next(self):
        """
        Returns the next record in the dataset as a list

        Returns:
            List, the next record in the dataset.
        """

        if self.offload_model is None:
            return [self._transform_md_to_output(t) for t in self._iterator.GetNextAsList()]
        data = [self._transform_md_to_tensor(t) for t in self._iterator.GetNextAsList()]
        if data:
            data = offload.apply_offload_iterators(data, self.offload_model)
        return [self._transform_tensor_to_output(t) for t in data]


class DummyIterator:
    """
    A DummyIterator only work when env MS_ROLE="MS_PSERVER" or MS_ROLE="MS_SCHED"
    """

    def __init__(self, dataset, mode, output_numpy=False):
        self.mode = mode
        self.shapes = dataset.output_shapes()
        self.types = dataset.output_types()
        self.col_names = dataset.get_col_names()
        self.fetched_first = False
        self.output_numpy = output_numpy

    def __get_tensor(self):
        """Get a next tensor."""
        tensor_row = []
        for np_shape, np_type in zip(self.shapes, self.types):
            input_np = np.zeros(np_shape, np_type)
            tensor = Tensor(input_np)
            if self.output_numpy:
                tensor_row.append(tensor.asnumpy())
            else:
                tensor_row.append(tensor)
        if self.mode == "dict":
            tensor_row = {col_name: tensor for col_name, tensor in zip(self.col_names, tensor_row)}
        return tensor_row

    def __iter__(self):
        return self

    def __next__(self):
        if not self.fetched_first:
            self.fetched_first = True
            return self.__get_tensor()
        raise StopIteration()
