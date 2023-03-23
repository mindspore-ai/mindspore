# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
"""
This file contains basic classes that help users do flexible dataset loading.
You can define your own dataset loading class, and use GeneratorDataset to help load data.
After declaring the dataset object, you can further apply dataset operations
(e.g. filter, skip, concat, map, batch) on it.
"""
import builtins
import math
import os
import signal
import time
import multiprocessing
from multiprocessing.util import Finalize
import queue
from functools import partial
import subprocess
import threading
import weakref
import platform
import psutil
import numpy as np

import mindspore._c_dataengine as cde

from mindspore.common import Tensor
from mindspore import log as logger

from .datasets import UnionBaseDataset, MappableDataset, Schema, to_list, _PythonMultiprocessing, _check_shm_usage
from . import samplers
from .queue import _SharedQueue
from .validators import check_generatordataset, check_numpyslicesdataset, check_paddeddataset
from ..core.config import get_enable_shared_mem, get_prefetch_size, get_multiprocessing_timeout_interval, \
    get_enable_watchdog, get_debug_mode
from ..core.datatypes import mstypelist_to_detypelist
from ..core.py_util_helpers import ExceptionHandler


def _iter_fn(dataset, num_samples):
    """
    Generator function wrapper for iterable dataset.
    """
    if num_samples is not None and num_samples != 0:
        ds_iter = iter(dataset)
        for _ in range(num_samples):
            try:
                val = next(ds_iter)
            except StopIteration:
                return
            # convert output tensors to ndarrays
            yield _convert_row(val)
    else:
        for val in dataset:
            # convert output tensors to ndarrays
            yield _convert_row(val)


def _generator_fn(generator, num_samples):
    """
    Generator function wrapper for generator function dataset.
    """
    if num_samples is not None and num_samples != 0:
        gen_iter = generator()
        for _ in range(num_samples):
            try:
                val = next(gen_iter)
            except StopIteration:
                return
            yield _convert_row(val)
    else:
        gen_iter = generator()
        for val in gen_iter:
            yield _convert_row(val)


def _cpp_sampler_fn(sample_ids, dataset):
    """
    Generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    for i in sample_ids:
        val = dataset[i]
        # convert output tensors to ndarrays
        yield _convert_row(val)


def _cpp_sampler_fn_mp(sample_ids, sample_fn):
    """
    Multiprocessing generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    return sample_fn.process(sample_ids)


def _fill_worker_indices(workers, indices, idx):
    """
    Worker index queue filler, fill worker index queue in round robin order.
    """
    num_worker = len(workers)
    while idx < len(indices):
        try:
            workers[idx % num_worker].put(indices[idx])
            idx += 1
        except queue.Full:
            break
    return idx


def _convert_row(row):
    """
    Convert Op return value to numpy, or keep as a dict (if already a dict)
    """

    # convert single item to np.array
    prim_type = (int, float, str, bytes, np.ndarray, Tensor)
    if isinstance(row, prim_type):
        if isinstance(row, Tensor):      # mindspore.Tensor
            item = row.asnumpy()
        else:
            item = np.array(row, copy=False)
            if item.dtype == 'object':
                raise TypeError("Data type of the input or its converted Numpy array is expected to be " \
                                "int or float or str, but got {}.".format(item.dtype))
        return tuple([item])

    if isinstance(row, dict):
        return tuple([row])

    value = []
    # convert each item to np.array
    idx = 0
    for x in row:
        idx += 1
        if isinstance(x, Tensor):      # mindspore.Tensor
            value.append(x.asnumpy())
        elif isinstance(x, dict):
            value.append(x)
        else:
            item = np.array(x, copy=False)
            if item.dtype == 'object':
                raise TypeError("Data type of {}th item of the input or its converted Numpy array is expected to be " \
                                "int or float or str, but got {}.".format(idx, item.dtype))
            value.append(item)
    return tuple(value)


class SamplerFn:
    """
    Multiprocessing or multithread generator function wrapper master process.
    """

    def __init__(self, dataset, num_worker, multi_process, max_rowsize):
        self.workers = []
        self.dataset = dataset
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.max_rowsize = max_rowsize
        self.need_join = False
        self.ppid = os.getpid()
        self.pids = []
        self.check_interval = get_multiprocessing_timeout_interval()  # the interval of check queue's size
        self._final_join = True

        # Event for end of epoch
        if multi_process is True:
            try:
                self.eof = multiprocessing.Event()
            except Exception:
                raise RuntimeError("Init multiprocessing.Event() failed, This might be caused by insufficient shm,"
                                   + " and the recommended shm size is at least 5 GB.")
        else:
            self.eof = threading.Event()
        # Create workers

        # get default queue size and adjust queuesize per worker if there are large # workers
        queue_size = get_prefetch_size()
        queue_size = min(queue_size, queue_size * 4 // num_worker)
        queue_size = max(2, queue_size)

        if multi_process and get_enable_shared_mem():
            _check_shm_usage(num_worker, queue_size, max_rowsize)
        count = multiprocessing.Value('i', 0)
        for _ in range(num_worker):
            if multi_process is True:
                try:
                    worker = _GeneratorWorkerMp(dataset, self.eof, max_rowsize, queue_size, self.ppid, count)
                except Exception:
                    raise RuntimeError("Init multiprocessing.Queue() failed, This might be caused by insufficient shm, "
                                       "and the recommended shm size is at least 5 GB.")
                worker.daemon = True
                # When multi processes fork a subprocess, the lock of the main process is copied to the subprocess,
                # which may cause deadlock. Therefore, the subprocess startup is performed in the initialization phase.
                # In this phase, the main process is not locked.
                worker.start()
                self.pids.append(worker.pid)
                self.need_join = True
            else:
                worker = _GeneratorWorkerMt(dataset, self.eof)
                worker.daemon = True
            self.workers.append(worker)
        self._launch_cleanup_worker(multi_process=multi_process)

    def process(self, indices):
        """
        The main process, start the child process or child thread, and fill the index queue.
        Get the result and return.
        """
        for w in self.workers:
            # Check whether the queue of the subprocess is empty.
            if not w.queue_empty():
                raise Exception("The queue of the subprocess is not empty.")
            # Start all workers
            if not w.is_alive():
                w.start()

        # Fill initial index queues
        idx_cursor = 0
        idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)

        # Fetch results
        for i in range(len(indices)):
            if self.eof.is_set():
                self._stop_subprocess()
                return
            if self.multi_process is True and not psutil.pid_exists(self.workers[i % self.num_worker].pid):
                self._stop_subprocess()
                return
            # Fetch result and put index
            try:
                # To avoid get timeout from queue, check the res_queue size.
                start_time = int(time.time())
                wait_count = 1
                while self.workers[i % self.num_worker].res_queue.empty():
                    if self.eof.is_set():
                        logger.warning("Generator receives a termination signal, stop waiting for data "
                                       "from subprocess.")
                        self._stop_subprocess()
                        return
                    time.sleep(0.1)
                    cost_time = int(time.time()) - start_time
                    if cost_time / self.check_interval >= wait_count:
                        wait_count += 1
                        logger.warning("It has been waiting for " + str(cost_time) + "s because the multi "
                                       "thread/process of the generator generates data had been hung by gil lock. "
                                       "Check whether the source of generator has an infinite loop operation or the "
                                       "output data is too large. You can also set the timeout interval by "
                                       "ds.config.set_multiprocessing_timeout_interval to adjust the output "
                                       "frequency of this log.")
                        pid = self.workers[i % self.num_worker].pid
                        logger.warning("Generator subprocess ID {} is stuck.".format(pid))
                        install_status, _ = subprocess.getstatusoutput("py-spy --version")
                        if install_status == 0:
                            stack = subprocess.getoutput("py-spy dump -p {} -l".format(pid))
                            logger.warning("Generator subprocess stack:\n{}".format(stack))
                        else:
                            logger.warning("Please `pip install py-spy` to get the stacks of the stuck process.")

                result = self.workers[i % self.num_worker].get()
                if isinstance(result, ExceptionHandler):
                    result.reraise()
            except queue.Empty:
                self._stop_subprocess()
                raise Exception("Generator worker process timeout.")
            except KeyboardInterrupt:
                self._stop_subprocess()
                raise Exception("Generator worker receives KeyboardInterrupt.")
            if self.eof.is_set():
                self._stop_subprocess()
                return
            if idx_cursor < len(indices):
                idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)
            yield _convert_row(result)

    def _launch_cleanup_worker(self, multi_process):
        """
        We need a extra thread and process if main process or subprocess was killed.

        Args:
            multi_process: Whether use multiprocess.
        """
        if multi_process is True and platform.system().lower() != 'windows':
            _clean_worker_func = _PythonMultiprocessing._clean_process  # pylint: disable=W0212
            self.cleaning_process = multiprocessing.Process(target=_clean_worker_func, args=(self.ppid, self.workers))
            self.cleaning_process.daemon = True
            self.cleaning_process.start()

            if get_enable_watchdog():
                self.eot = threading.Event()
                self.watch_dog = threading.Thread(target=_PythonMultiprocessing._watch_dog,  # pylint: disable=W0212
                                                  args=(self.eot, self.workers + [self.cleaning_process]))
                self.watch_dog.daemon = True
                self.watch_dog.start()

                if self._final_join is True:
                    self._jointhread = Finalize(
                        self.watch_dog, self._finalize_join,
                        args=(weakref.ref(self.watch_dog), self.eot),
                        exitpriority=-5
                    )

    def _stop_subprocess(self):
        """Only the main process can call join."""
        if self.need_join is True and self.ppid == os.getpid():
            if hasattr(self, 'eof') and self.eof is not None and not self.eof.is_set():
                self.eof.set()
            self.need_join = False
            for w in self.workers:
                if self.multi_process is True and hasattr(w, '_closed') and w._closed is False:  # pylint: disable=W0212
                    try:
                        w.join()
                    except Exception:  # pylint: disable=W0703
                        # Block all errors when join
                        continue
            self._abort_watchdog()

    def _abort_watchdog(self):
        if hasattr(self, 'eot') and self.eot is not None and not self.eot.is_set():
            self.eot.set()
        if hasattr(self, 'cleaning_process') and self.cleaning_process is not None:
            _PythonMultiprocessing._terminate_processes([self.cleaning_process])  # pylint: disable=W0212

    @classmethod
    def _finalize_join(cls, twr, eot):
        thread = twr()
        if thread is not None:
            if eot is not None and not eot.is_set():
                eot.set()
            thread.join()

    def __del__(self):
        try:
            self._stop_subprocess()
        except TypeError:
            pass

    def __deepcopy__(self, memodict, exclude=()):
        self.__init__(self.dataset, self.num_worker, self.multi_process, self.max_rowsize)


def _subprocess_handle(eof, signum, frame):
    threading.Thread(target=eof.set()).start()


def _ignore_sigint(is_multiprocessing):
    """
    We need to ignore sigint signal here so subprocesses can exit normally and clear.
    """
    if is_multiprocessing:
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _main_process_already_exit(eof, is_multiprocessing, idx_queue, result_queue, ppid):
    """
    Judge whether main process already exit.
    """
    if eof.is_set() or (is_multiprocessing and platform.system().lower() != 'windows' and
                        not _PythonMultiprocessing.is_process_alive(ppid)):
        if is_multiprocessing:
            idx_queue.cancel_join_thread()
            result_queue.cancel_join_thread()
        return True
    return False


def _generator_worker_loop(dataset, idx_queue, result_queue, eof, is_multiprocessing, ppid=-1):
    """
    Multithread or multiprocess generator worker process loop.
    """
    if is_multiprocessing:
        signal.signal(signal.SIGTERM, partial(_subprocess_handle, eof))
    while True:
        _ignore_sigint(is_multiprocessing=is_multiprocessing)

        # Fetch index, block
        try:
            idx = idx_queue.get(timeout=1)
        except queue.Empty:
            if _main_process_already_exit(eof, is_multiprocessing, idx_queue, result_queue, ppid) is True:
                return
            # If end-of-file (eof) is not set, continue to get data from idx_queue
            continue
        if idx is None:
            # When the queue is out of scope from master process, a None item can be fetched from the queue.
            # Upon receiving None, worker process should check if eof is set.
            if not eof.is_set():
                raise Exception("")
            return
        if eof.is_set():
            if is_multiprocessing:
                idx_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
            return
        # Fetch data, any exception from __getitem__ will terminate worker and timeout master process
        try:
            result = dataset[idx]
        except Exception:  # pylint: disable=broad-except
            result = ExceptionHandler(where="in GeneratorDataset worker process")
        # Send data, block
        while True:
            try:
                result_queue.put(result, timeout=5)
            except queue.Full:
                if _main_process_already_exit(eof, is_multiprocessing, idx_queue, result_queue, ppid) is True:
                    return
                # If eof is not set, continue to put data to result_queue
                continue
            break
        del result, idx


class _GeneratorWorkerMt(threading.Thread):
    """
    Worker process for multi-thread Generator.
    """

    def __init__(self, dataset, eof):
        self.idx_queue = queue.Queue(16)
        self.res_queue = queue.Queue(16)
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, False))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty")
            return False
        return True


class _GeneratorWorkerMp(multiprocessing.Process):
    """
    Worker process for multiprocess Generator.
    """

    def __init__(self, dataset, eof, max_rowsize, queue_size, ppid, count):
        self.idx_queue = multiprocessing.Queue(queue_size)
        if get_enable_shared_mem():
            self.res_queue = _SharedQueue(queue_size, count, max_rowsize=max_rowsize)
        else:
            self.res_queue = multiprocessing.Queue(queue_size)
        self.idx_queue._joincancelled = True  # pylint: disable=W0212
        self.res_queue._joincancelled = True  # pylint: disable=W0212
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, True, ppid))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        # Relax 10s to 30s, since it sometimes will cause "Generator worker process timeout"
        # when we run too many iterators with infinite epoch(num_epoch=-1)
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty.")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty.")
            return False
        return True

    def __del__(self):
        # del all the Queue & SharedQueue when the iter had been deleted from ITERATORS_LIST
        del self.idx_queue
        del self.res_queue


class GeneratorDataset(MappableDataset, UnionBaseDataset):
    """
    A source dataset that generates data from Python by invoking Python data source each epoch.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        source (Union[Callable, Iterable, Random Accessible]):
            A generator callable object, an iterable Python object or a random accessible Python object.
            Callable source is required to return a tuple of NumPy arrays as a row of the dataset on source().next().
            Iterable source is required to return a tuple of NumPy arrays as a row of the dataset on
            iter(source).next().
            Random accessible source is required to return a tuple of NumPy arrays as a row of the dataset on
            source[idx].
        column_names (Union[str, list[str]], optional): List of column names of the dataset. Default: None. Users are
            required to provide either column_names or schema.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset. Default: None.
            If provided, sanity check will be performed on generator output.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by mindspore.dataset.Schema are acceptable.
            Default: None.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: None, all images.
        num_parallel_workers (int, optional): Number of worker threads/subprocesses used to
            fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            Default: None, expected order behavior shown in the table below.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required. Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            Random accessible input is required. When this argument is specified, `num_samples` reflects the maximum
            sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument must be specified only
            when `num_shards` is also specified. Random accessible input is required.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: True.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True. Default: 6 MB.

    Raises:
        RuntimeError: If source raises an exception during execution.
        RuntimeError: If len of column_names does not match output len of source.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If sampler and shuffle are specified at the same time.
        ValueError: If sampler and sharding are specified at the same time.
        ValueError: If `num_shards` is specified but shard_id is None.
        ValueError: If shard_id is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Note:
        - If you configure `python_multiprocessing=True (default: True)` and `num_parallel_workers>1 (default: 1)`
          indicates that the multi-process mode is started for data load acceleration. At this time, as the dataset
          iterates, the memory consumption of the subprocess will gradually increase, mainly because the subprocess
          of the user-defined dataset obtains the member variables from the main process in the Copy On Write way.
          Example: If you define a dataset with `__ init__` function which contains a large number of member variable
          data (for example, a very large file name list is loaded during the dataset construction) and uses the
          multi-process mode, which may cause the problem of OOM (the estimated total memory usage is:
          `(num_parallel_workers+1) * size of the parent process` ). The simplest solution is to replace python objects
          (such as list/dict/int/float/string) with non referenced data types
          (such as Pandas, Numpy or PyArrow objects) for member variables, or configure `python_multiprocessing=False`
          multi-threading mode is used.
        - Input `source` accepts user-defined Python functions (PyFuncs), Do not add network computing operators from
          mindspore.nn and mindspore.ops or others into this `source` .
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> import numpy as np
        >>>
        >>> # 1) Multidimensional generator function as callable input.
        >>> def generator_multidimensional():
        ...     for i in range(64):
        ...         yield (np.array([[i, i + 1], [i + 2, i + 3]]),)
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multidimensional, column_names=["multi_dimensional_data"])
        >>>
        >>> # 2) Multi-column generator function as callable input.
        >>> def generator_multi_column():
        ...     for i in range(64):
        ...         yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["col1", "col2"])
        >>>
        >>> # 3) Iterable dataset as iterable input.
        >>> class MyIterable:
        ...     def __init__(self):
        ...         self._index = 0
        ...         self._data = np.random.sample((5, 2))
        ...         self._label = np.random.sample((5, 1))
        ...
        ...     def __next__(self):
        ...         if self._index >= len(self._data):
        ...             raise StopIteration
        ...         else:
        ...             item = (self._data[self._index], self._label[self._index])
        ...             self._index += 1
        ...             return item
        ...
        ...     def __iter__(self):
        ...         self._index = 0
        ...         return self
        ...
        ...     def __len__(self):
        ...         return len(self._data)
        >>>
        >>> dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])
        >>>
        >>> # 4) Random accessible dataset as random accessible input.
        >>> class MyAccessible:
        ...     def __init__(self):
        ...         self._data = np.random.sample((5, 2))
        ...         self._label = np.random.sample((5, 1))
        ...
        ...     def __getitem__(self, index):
        ...         return self._data[index], self._label[index]
        ...
        ...     def __len__(self):
        ...         return len(self._data)
        >>>
        >>> dataset = ds.GeneratorDataset(source=MyAccessible(), column_names=["data", "label"])
        >>>
        >>> # list, dict, tuple of Python is also random accessible
        >>> dataset = ds.GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])
    """

    @check_generatordataset
    def __init__(self, source, column_names=None, column_types=None, schema=None, num_samples=None,
                 num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
                 python_multiprocessing=True, max_rowsize=6):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
        if isinstance(source, builtins.zip):
            # Although zip is iterable, it does not have the feature of repeated iteration, so pass it to the array.
            self.source = [item for item in source]
        else:
            self.source = source
        self.prepared_source = None  # source to be sent to C++
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True:
            self.num_parallel_workers = 1
            logger.warning(
                "Input 'source' of 'GeneratorDataset' includes network computing operators like in mindspore.nn, "
                "mindspore.ops, mindspore.numpy module and etc, which do not support multi-thread compiling, recommend"
                " to replace it with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' "
                "into 1.")

        if platform.system().lower() == 'windows' and num_parallel_workers > 1 and python_multiprocessing:
            logger.warning("Python multiprocessing is not supported on Windows platform.")
        self.python_multiprocessing = python_multiprocessing if platform.system().lower() != 'windows' else False
        if self.python_multiprocessing and get_debug_mode():
            logger.warning("Python multiprocessing is not supported in debug mode."
                           " Ignoring Python multiprocessing for GeneratorDataset.")
            self.python_multiprocessing = False

        self.column_names = to_list(column_names)

        if column_types is not None:
            self.column_types = mstypelist_to_detypelist(column_types)
        else:
            self.column_types = []

        self.schema = schema
        if schema is not None:
            self.schema = schema
            if not isinstance(schema, Schema):
                self.schema = Schema(schema)
        # Move get dataset_size by len from parse to here, because self.source will
        # lose attribution of '__len__' after deepcopy.
        self.source_len = -1  # unknown
        if hasattr(self.source, "__len__"):
            self.source_len = len(self.source)

            # if user defined sampler, update the self.source_len
            if isinstance(self.sampler, samplers.Sampler) or hasattr(self.sampler, "__iter__"):
                self.source_len = len(list(sampler))

        self.max_rowsize = max_rowsize
        self.sample_fn = None

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        new_op = self.__safe_deepcopy__(memodict, exclude=("source", "__transfer_dataset__"))

        sample_fn = None
        if new_op.sampler is not None and hasattr(self.source, "__getitem__"):
            # The reason why there is a try catch here is because when the new op is being constructed with shared
            # memory enabled, there will be an exception thrown if there is not enough shared memory available
            if self.source_len == -1:
                raise RuntimeError("Attempt to construct a random access dataset, '__len__' method is required!")
            try:
                if new_op.num_parallel_workers > 1:
                    self.__validate_memory_usage()

                    sample_fn = SamplerFn(self.source, new_op.num_parallel_workers, self.python_multiprocessing,
                                          self.max_rowsize)
                    new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn_mp(sample_ids, sample_fn))
                else:
                    new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn(sample_ids, self.source))
                new_op.sample_fn = sample_fn
            except RuntimeError as e:
                raise Exception(str(e))
        else:
            try:
                new_op.sampler = None
                new_op.sample_fn = sample_fn
                new_op.source_len = min(new_op.source_len,
                                        new_op.num_samples) if new_op.num_samples != 0 else new_op.source_len
                iter(self.source)
            except TypeError:
                # Use generator function if input callable
                new_op.prepared_source = (lambda: _generator_fn(self.source, new_op.num_samples))
            else:
                # Use iterator function if input is iterable
                # Random accessible input is also iterable
                new_op.prepared_source = (lambda: _iter_fn(self.source, new_op.num_samples))

        return new_op

    def is_shuffled(self):
        if self.sampler:
            return self.sampler.is_shuffled()
        return False

    def is_sharded(self):
        if self.sampler:
            return self.sampler.is_sharded()
        return False

    def split(self, sizes, randomize=True):
        if hasattr(self.source, "__getitem__"):
            # If the source has __getitem__ attribute, call the split method of MappableDataset.
            # Otherwise, call the split method of Dataset.
            return super().split(sizes, randomize)
        return super(MappableDataset, self).split(sizes, randomize)

    def parse(self, children=None):
        if self.schema is None:
            return cde.GeneratorNode(self.prepared_source, self.column_names, self.column_types, self.source_len,
                                     self.sampler, self.num_parallel_workers)
        schema = self.schema
        if isinstance(schema, Schema):
            schema = self.schema.cpp_schema
        return cde.GeneratorNode(self.prepared_source, schema, self.source_len, self.sampler,
                                 self.num_parallel_workers)

    def __validate_memory_usage(self):
        """
        Check memory usage when mulit-processing mode, when 85% prompt warning and 100% raise error.
        """
        if self.python_multiprocessing:
            # if use num_parallel_workers is to large when python_multiprocessing=True which would cause
            # OOM error get the num_shards
            valid_num_shards = 1
            if isinstance(self.sampler, samplers.DistributedSampler):
                valid_num_shards = self.sampler.num_shards
            elif self.num_shards is not None:
                valid_num_shards = self.num_shards

            # get process memory usage
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss
            sys_memory_free = psutil.virtual_memory().free

            total_memory_maybe_used = process_memory * self.num_parallel_workers * valid_num_shards
            if total_memory_maybe_used / sys_memory_free > 0.85:
                valid_num_worker = math.floor(sys_memory_free * 0.85 / valid_num_shards / process_memory)
                valid_num_worker = 1 if valid_num_worker <= 0 else valid_num_worker
                info = "GeneratorDataset's num_parallel_workers: {} is too large which may cause a lot of memory " \
                       "occupation (>85%) or out of memory(OOM) during multiprocessing. Therefore, it is recommended " \
                       "to reduce num_parallel_workers to {} or smaller.".format(self.num_parallel_workers,
                                                                                 valid_num_worker)
                logger.warning(info)


class _NumpySlicesDataset:
    """
    Mainly for dealing with several kinds of formats of Python data, and return one row each time.
    """

    def __init__(self, data, column_list=None):
        self.column_list = None
        # Convert dict data into tuple
        if isinstance(data, dict):
            data = self.process_dict(data)

        if isinstance(data, tuple):
            self.data = data
        else:
            self.data = (data,)

        # check whether the data length in each column is equal
        data_len = [len(data_item) for data_item in self.data]
        if data_len[1:] != data_len[:-1]:
            raise ValueError("Data length in each column is not equal.")

        # Init column_name
        if column_list is not None:
            self.column_list = column_list
        elif self.column_list is None:
            self.column_list = []
            column_num = len(self.data)
            for i in range(column_num):
                self.column_list.append("column_" + str(i))

    def __getitem__(self, index):
        data_row = [d[index] for d in self.data]
        data_res = tuple(data_row)
        return data_res

    def __len__(self):
        return len(self.data[0])

    def process_dict(self, input_data):
        """
        Convert the dict like data into tuple format, when input is a tuple of dicts then compose it into a dict first.
        """
        # Convert pandas like dict(has "values" column) into General dict
        data_keys = list(input_data.keys())
        data_col = input_data[data_keys[0]]
        if hasattr(data_col, "values"):
            new_dict = {}
            for key in data_keys:
                item1 = input_data.pop(key)
                new_dict[key] = item1.values
            input_data = new_dict

        # Convert the data in dict into tuple
        data = ()
        keys = list(input_data.keys())
        self.column_list = keys
        for key in keys:
            value = input_data[key]
            data = data + (list(value),)

        return data


class NumpySlicesDataset(GeneratorDataset):
    """
    Creates a dataset with given data slices, mainly for loading Python data into dataset.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        data (Union[list, tuple, dict]) Input of given data. Supported data types include: list, tuple, dict and other
            NumPy formats. Input data will be sliced along the first dimension and generate additional rows, if input is
            list, there will be one column in each row, otherwise there tends to be multi columns. Large data is not
            recommended to be loaded in this way as data is loading into memory.
        column_names (list[str], optional): List of column names of the dataset. Default: None. If column_names is not
            provided, the output column names will be named as the keys of dict when the input data is a dict,
            otherwise they will be named like column_0, column_1 ...
        num_samples (int, optional): The number of samples to be included in the dataset. Default: None, all samples.
        num_parallel_workers (int, optional): Number of worker subprocesses used to
            fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: None, expected order behavior shown in the table below.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset.
            Default: None, expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument must be specified only
            when `num_shards` is also specified.

    Note:
        - This dataset can take in a `sampler` . `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Raises:
        RuntimeError: If len of column_names does not match output len of data.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If sampler and shuffle are specified at the same time.
        ValueError: If sampler and sharding are specified at the same time.
        ValueError: If `num_shards` is specified but shard_id is None.
        ValueError: If shard_id is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> # 1) Input data can be a list
        >>> data = [1, 2, 3]
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1"])
        >>>
        >>> # 2) Input data can be a dictionary, and column_names will be its keys
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> dataset = ds.NumpySlicesDataset(data=data)
        >>>
        >>> # 3) Input data can be a tuple of lists (or NumPy arrays), each tuple element refers to data in each column
        >>> data = ([1, 2], [3, 4], [5, 6])
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1", "column_2", "column_3"])
        >>>
        >>> # 4) Load data from CSV file
        >>> import pandas as pd
        >>> df = pd.read_csv(filepath_or_buffer=csv_dataset_dir[0])
        >>> dataset = ds.NumpySlicesDataset(data=dict(df), shuffle=False)
    """

    @check_numpyslicesdataset
    def __init__(self, data, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None):
        dataset = _NumpySlicesDataset(data, column_names)
        super().__init__(dataset, column_names=dataset.column_list, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class _PaddedDataset:
    """
    Mainly for combining false samples provided by users into a dataset.

    Args:
        padded_samples (list(dict)): Data provided by user to be added to the initial Dataset.
    """

    def __init__(self, padded_samples):
        self.column_names = list(padded_samples[0].keys())
        self.padded_samples = padded_samples

    def __getitem__(self, item):
        return (self.padded_samples[item][key] for key in self.column_names)

    def __len__(self):
        return len(self.padded_samples)


class PaddedDataset(GeneratorDataset):
    """
    Creates a dataset with filler data provided by user.

    Mainly used to add to the original dataset and assign it to the corresponding shard.

    Args:
        padded_samples (list(dict)): Samples provided by user.

    Raises:
        TypeError: If padded_samples is not an instance of list.
        TypeError: If the element of padded_samples is not an instance of dict.
        ValueError: If the padded_samples is empty.

    Examples:
        >>> import numpy as np
        >>> data = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)}]
        >>> dataset = ds.PaddedDataset(padded_samples=data)
    """

    @check_paddeddataset
    def __init__(self, padded_samples):
        dataset = _PaddedDataset(padded_samples)
        super().__init__(dataset, column_names=dataset.column_names, num_shards=None, shard_id=None, shuffle=False)
        self._dataset_size = len(dataset.padded_samples)
        self.padded_samples = padded_samples
