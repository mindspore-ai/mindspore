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
import atexit
import json
import os
import queue
import signal
import threading
import weakref
from functools import wraps
import numpy as np

import mindspore._c_dataengine as cde
from mindspore.common.tensor import Tensor, np_types
import mindspore.dataset as ds
from mindspore.dataset.engine import offload
from mindspore.dataset.core.config import get_debug_mode

from mindspore import log as logger
from ..core.py_util_helpers import ExceptionHandler

_ITERATOR_CLEANUP = False


def _set_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = True


def _unset_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = False


def check_iterator_cleanup():
    return _ITERATOR_CLEANUP


ITERATORS_LIST = []


def _cleanup():
    """Release all the Iterator."""
    _set_iterator_cleanup()
    for itr_ref in reversed(ITERATORS_LIST):
        itr = itr_ref()
        if itr is not None:
            itr.release()


def _cleanup_the_iterators_if_created(method):
    """Release the iterators which is new created by the method"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        original_iterators = deepcopy(ITERATORS_LIST)

        result = method(self, *args, **kwargs)

        # it is used to attribute function like: dataset_size / output_shapes / output_types and
        # it is a GeneratorDataset with two stage pipeline. The first pipeline will create a new iterator
        # which need to be released after dataset_size / output_shapes / output_types end.
        # 1. find the iterators which are started by dataset_size / output_shapes / output_types with two stage pipeline
        iterators_to_be_released = []
        for index, item in enumerate(ITERATORS_LIST):
            if item not in original_iterators:
                iterators_to_be_released.append(index)

        # 2. release the iterators
        for index in reversed(iterators_to_be_released):
            itr = ITERATORS_LIST[index]()
            if itr is not None:
                itr.release()

        return result
    return wrapper


def __convert_python(obj, to_numpy, _do_copy):
    """
    Attempts to recursively convert a python object to Numpy array(s) or tensor(s).

    Args:
        obj (any): the python object to be converted
        to_numpy (bool): If True, convert primitive types to NumPy array. If False, convert to Tensor.
                         (return the obj if type isn't supported)
    """
    if isinstance(obj, (int, float, bool, str, np.ndarray, np.str_, np.bytes_, *np_types)):
        # error out if array is of unsupported type
        if isinstance(obj, np.ndarray) and obj.dtype not in np_types and obj.dtype.kind not in ('U', 'S'):
            raise TypeError("A NumPy array of unsupported type detected: " + str(obj.dtype) +
                            ".\nSupported types are: " + str('\n'.join(map(str, (*np_types, np.str_, np.bytes_)))) +
                            ".")
        if to_numpy:
            return np.array(obj, copy=_do_copy)

        # don't convert np.str_ or np.bytes_ to ms.Tensor
        np_array = np.asarray(obj)
        if np.issubdtype(np_array.dtype, np.str_) or np.issubdtype(np_array.dtype, np.bytes_):
            return np_array

        if _do_copy:
            return Tensor(np_array)
        return Tensor.from_numpy(np_array)
    if isinstance(obj, dict):
        return {key: __convert_python(val, to_numpy, _do_copy) for key, val in obj.items()}
    if isinstance(obj, tuple):
        return tuple(__convert_python(item, to_numpy, _do_copy) for item in obj)
    if isinstance(obj, list):
        return [__convert_python(item, to_numpy, _do_copy) for item in obj]
    # if we can't convert it to Tensor, return the object as is
    if _do_copy:
        return deepcopy(obj)
    return obj


def _transform_md_to_output(t, _output_numpy, _do_copy):
    if _output_numpy:
        if t.type().is_python():
            return __convert_python(t.as_python(), True, _do_copy)
        return t.as_array()
    return _transform_md_to_tensor(t, _do_copy)


def _transform_md_to_tensor(t, _do_copy):
    """transform md tensor type to ms tensor"""
    if t.type().is_python():
        return __convert_python(t.as_python(), False, _do_copy)
    array = t.as_array()
    # don't convert np.str_ or np.bytes_ to ms.Tensor
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return array
    if _do_copy:
        return Tensor(array)
    return Tensor.from_numpy(array)


def _transform_tensor_to_output(t, _output_numpy):
    if _output_numpy:
        return t.asnumpy()
    return t


def _convert_tuple_data(queue_in, queue_out, offload_model, _output_numpy, _do_copy, thread_event):
    """
    Convert data on tuple iterator.
    """
    while True:
        try:
            if thread_event.is_set():
                return
            item = queue_in.get(timeout=1)
            if offload_model is None:
                queue_out.put([_transform_md_to_output(t, _output_numpy, _do_copy) for t in item])
                if not item:
                    break
                continue

            data = [_transform_md_to_tensor(t, _do_copy) for t in item]
            if data:
                data = offload.apply_offload_iterators(data, offload_model)
            queue_out.put([_transform_tensor_to_output(t, _output_numpy) for t in data])
            if not item:
                break
        except queue.Empty:
            continue
        except Exception:  # pylint: disable=broad-except
            result = ExceptionHandler()
            queue_out.put(result)
            break
    thread_event.set()


def _convert_dict_data(queue_in, queue_out, offload_model, _output_numpy, _do_copy, thread_event, col_names):
    """
    Convert data on dict iterator.
    """
    while True:
        try:
            if thread_event.is_set():
                return
            item = queue_in.get(timeout=1)
            if offload_model is None:
                queue_out.put({k: _transform_md_to_output(t, _output_numpy, _do_copy) for k, t in item})
                if not item:
                    break
                continue
            data = [_transform_md_to_tensor(t, _do_copy) for t in item]
            if data:
                data = offload.apply_offload_iterators(data, offload_model)
                # Create output dictionary after offload
                out_data = {}
                for i, col in enumerate(col_names):
                    out_data[col] = _transform_tensor_to_output(data[i], _output_numpy)
                data = out_data
            if not item:
                break
            queue_out.put(data)
        except queue.Empty:
            continue
        except Exception:  # pylint: disable=broad-except
            result = ExceptionHandler()
            queue_out.put(result)
            break
    thread_event.set()


class Iterator:
    """
    General Iterator over a dataset.

    Attributes:
        dataset: Dataset to be iterated over
    """

    def __init__(self, dataset, num_epochs=-1, output_numpy=False, do_copy=False):
        self._col_names = None

        # create a copy of tree and work on it.
        self.__ori_dataset = dataset

        self.ir_tree, self.dataset = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        if dataset.get_init_step() == 0:
            init_step = 0
            dataset_size = -1
        else:
            init_step = dataset.get_init_step()
            dataset_size = dataset.get_dataset_size()
        if get_debug_mode():
            if dataset.get_init_step() != 0:
                logger.warning("Dataset init step will be ignored in debug mode.")
            consumer = cde.PythonPullBasedIteratorConsumer(num_epochs)
            consumer.Init(self.ir_tree)
        else:
            consumer = cde.PythonIteratorConsumer(num_epochs)
            consumer.Init(self.ir_tree, init_step, dataset_size)
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
        self.parallel_convert = ds.config.get_iterator_mode()["parallel_convert"]
        if self.parallel_convert:
            # The variable "tick" ensures that the thread is only started on the first iteration
            self.tick = True
            self.thread_convert = None
            self.queue_in = queue.Queue(3)
            self.queue_out = queue.Queue(3)
            self.thread_event = None
            self.enable_get_next_data = True
            atexit.register(self.__class__.terminate, weakref.ref(self))

    def __iter__(self):
        return self

    @staticmethod
    def terminate(ref):
        """
        Interrupt the convert subthread
        """
        self = ref()
        if self is None:
            return
        if hasattr(self, "parallel_convert"):
            if self.parallel_convert:
                if self.thread_event is not None and self.thread_convert is not None and \
                    not self.thread_event.is_set():
                    self.thread_event.set()
                    self.thread_convert.join()

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
        if hasattr(self, "parallel_convert"):
            if self.parallel_convert:
                if self.thread_event is not None and self.thread_convert is not None and \
                    not self.thread_event.is_set():
                    self.thread_event.set()
                    self.thread_convert.join()
        self.release()

    @abstractmethod
    def _get_next(self):
        raise RuntimeError("Calling base class Iterator's get_next is invalid.")

    @abstractmethod
    def _parallel_transformation_iteration(self):
        raise RuntimeError("Calling base class Iterator's parallel_transformation_iteration is invalid.")

    def serial_conversion_iteration(self):
        """
        Fetch data to serial conversion
        """
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

    def __next__(self):
        if not self._runtime_context:
            logger.warning("Iterator does not have a running C++ pipeline." +
                           "It might because Iterator stop() had been called, or C++ pipeline crashed silently.")
            raise RuntimeError("Iterator does not have a running C++ pipeline.")

        from mindspore.profiler import mstx  # pylint: disable=import-outside-toplevel
        range_id = mstx.range_start('dataloader', None)
        out = self._parallel_transformation_iteration() if self.parallel_convert else self.serial_conversion_iteration()
        mstx.range_end(range_id)
        return out

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

    def _reset(self, step, dataset_size):
        """
        Reset the iterator to the given step number and epoch number.

        Args:
            step (int): Global step number
            dataset_size (int): The number of steps that one epoch has.
        """
        self._iterator.Reset(step, dataset_size)
        if self.parallel_convert:
            while not self.queue_in.empty():
                self.queue_in.get()
            while not self.queue_out.empty():
                self.queue_out.get()
            if self.thread_event is not None:
                if self.thread_event.is_set():
                    self.thread_event.clear()
                    self.tick = True
                else:
                    self.thread_event.set()
                    self.thread_convert.join()
            self.tick = True
            self.enable_get_next_data = True


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
                return {k: _transform_md_to_output(t, self._output_numpy, self._do_copy) for k, t in
                        self._iterator.GetNextAsMap().items()}
            data = [_transform_md_to_tensor(t, self._do_copy) for t in self._iterator.GetNextAsList()]
            if data:
                data = offload.apply_offload_iterators(data, self.offload_model)
                # Create output dictionary after offload
                out_data = {}
                for i, col in enumerate(self.get_col_names()):
                    out_data[col] = _transform_tensor_to_output(data[i], self._output_numpy)
                data = out_data
            return data

        except RuntimeError as err:
            # maybe "Out of memory" / "MemoryError" error
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.critical("Memory error occurred, process will exit.")
                os.kill(os.getpid(), signal.SIGKILL)
            raise err

    def _parallel_transformation_iteration(self):
        """
        Launch child thread to convert tensor.
        """
        if self.tick:
            self.thread_event = threading.Event()
            self.thread_convert = threading.Thread(target=_convert_dict_data,
                                                   name="Convert_dict_data",
                                                   args=(self.queue_in, self.queue_out, self.offload_model,
                                                         self._output_numpy, self._do_copy, self.thread_event,
                                                         self.get_col_names()),
                                                   daemon=True)
            self.thread_convert.start()
            self.tick = False
        while True:
            if self.thread_event.is_set() and self.queue_out.qsize() == 0:
                self.tick = True
                self.enable_get_next_data = True
                raise StopIteration
            try:
                if not self.queue_in.full() and self.enable_get_next_data:
                    if self.offload_model is None:
                        item = self._iterator.GetNextAsMap().items()
                        if not item:
                            self.enable_get_next_data = False
                        self.queue_in.put(item)
                    else:
                        item = self._iterator.GetNextAsList()
                        if not item:
                            self.enable_get_next_data = False
                        self.queue_in.put(item)
                data = self.queue_out.get(timeout=0.00001)
                if not data:
                    continue
                if isinstance(data, ExceptionHandler):
                    if data.except_msg.find("Out of memory") >= 0 or data.except_msg.find("MemoryError") >= 0:
                        logger.critical("Memory error occurred, process will exit.")
                        os.kill(os.getpid(), signal.SIGKILL)
                    data.reraise()
                return data
            except queue.Empty:
                continue
            except Exception as err_info:
                self.thread_event.set()
                self.thread_convert.join()
                raise err_info


class TupleIterator(Iterator):
    """
    The derived class of Iterator with list type.
    """

    def __init__(self, dataset, columns=None, num_epochs=-1, output_numpy=False, do_copy=False):
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
            return [_transform_md_to_output(t, self._output_numpy, self._do_copy) for t in
                    self._iterator.GetNextAsList()]
        data = [_transform_md_to_tensor(t, self._do_copy) for t in self._iterator.GetNextAsList()]
        if data:
            data = offload.apply_offload_iterators(data, self.offload_model)
        return [_transform_tensor_to_output(t, self._output_numpy) for t in data]

    def _parallel_transformation_iteration(self):
        """
        Launch child thread to convert tensor.
        """
        if self.tick:
            self.thread_event = threading.Event()
            self.thread_convert = threading.Thread(target=_convert_tuple_data,
                                                   name="Convert_tuple_data",
                                                   args=(self.queue_in, self.queue_out, self.offload_model,
                                                         self._output_numpy, self._do_copy, self.thread_event),
                                                   daemon=True)
            self.thread_convert.start()
            self.tick = False
        while True:
            if self.thread_event.is_set() and self.queue_out.qsize() == 0:
                self.tick = True
                self.enable_get_next_data = True
                raise StopIteration
            try:
                if not self.queue_in.full() and self.enable_get_next_data:
                    item = self._iterator.GetNextAsList()
                    if not item:
                        self.enable_get_next_data = False
                    self.queue_in.put(item)
                data = self.queue_out.get(timeout=0.00001)
                if not data:
                    continue
                if isinstance(data, ExceptionHandler):
                    data.reraise()
                return data
            except queue.Empty:
                continue
            except Exception as err_info:
                self.thread_event.set()
                self.thread_convert.join()
                raise err_info


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
            tensor_row = dict(zip(self.col_names, tensor_row))
        return tensor_row

    def __iter__(self):
        return self

    def __next__(self):
        if not self.fetched_first:
            self.fetched_first = True
            return self.__get_tensor()
        raise StopIteration()
