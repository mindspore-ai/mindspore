# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Built-in iterators.
"""
from abc import abstractmethod
import os
import signal
import weakref
import numpy as np

from mindspore.common.tensor import Tensor
import mindspore._c_dataengine as cde

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
    for itr_ref in ITERATORS_LIST:
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
        self.ori_dataset = dataset

        self.ir_tree, self.dataset = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        consumer = cde.PythonIteratorConsumer(num_epochs)
        consumer.Init(self.ir_tree)
        self._runtime_context.AssignConsumer(consumer)
        self._iterator = self._runtime_context.GetConsumer()

        self._transform_tensor = lambda t: t.as_array()
        if not output_numpy:
            if do_copy:
                self._transform_tensor = lambda t: Tensor(t.as_array())
            else:
                self._transform_tensor = lambda t: Tensor.from_numpy(t.as_array())
        self._index = 0

        # todo remove next when ContextManager is done
        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()
        #######

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

        data = self._get_next()
        if not data:
            if self._index == 0:
                logger.warning("No records available.")
            if self.ori_dataset.dataset_size is None:
                self.ori_dataset.dataset_size = self._index
            raise StopIteration
        self._index += 1
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
            self._getters()
        return self._col_names


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
            return {k: self._transform_tensor(t) for k, t in self._iterator.GetNextAsMap().items()}
        except RuntimeError as err:
            ## maybe "Out of memory" / "MemoryError" error
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.error("Memory error occurred, process will exit.")
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
            # todo: move next to IR
            dataset = dataset.project(columns)
        super().__init__(dataset, num_epochs, output_numpy, do_copy)

    def _get_next(self):
        """
        Returns the next record in the dataset as a list

        Returns:
            List, the next record in the dataset.
        """

        return [self._transform_tensor(t) for t in self._iterator.GetNextAsList()]


class DummyIterator:
    """
    A DummyIterator only work when env MS_ROLE="MS_PSERVER" or MS_ROLE="MS_SCHED"
    """

    def __init__(self, dataset, mode):
        self.mode = mode
        self.shapes = dataset.output_shapes()
        self.types = dataset.output_types()
        self.fetched_first = False

    def __get_tensor(self):
        tensor_row = []
        for np_shape, np_type in zip(self.shapes, self.types):
            input_np = np.zeros(np_shape, np_type)
            tensor = Tensor(input_np)
            tensor_row.append(tensor)
        return tensor_row

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "tuple":
            if not self.fetched_first:
                self.fetched_first = True
                return self.__get_tensor()
        raise StopIteration()
