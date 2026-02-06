# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Worker module."""

import os
import random
import sys
import traceback
from queue import Empty

import numpy as np

import mindspore
import mindspore._c_dataengine as cde
from mindspore import log as logger
from . import WORKER_TIME_OUT
from .bit_generator import seed_sequence

WORKER_INFO_LOCAL = None


class ResumeIterationFlag:
    """
    Flag for resume iteration.
    """


class WorkerInfo:
    """
    Worker information.
    """

    _initialized = False

    def __init__(self, id, num_workers, seed, dataset):  # pylint: disable=redefined-builtin
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset
        self._initialized = True

    def __setattr__(self, key, value):
        if self._initialized:
            raise RuntimeError("Cannot modify the attributes of WorkerInfo object after initialization.")
        return super().__setattr__(key, value)

    def __repr__(self):
        return (
            f"WorkerInfo: {{id: {self.id}, num_workers: {self.num_workers}, "
            f"seed: {self.seed}, dataset: {self.dataset}}}"
        )


class KeyErrorMsg(str):
    """
    Key error message.
    """

    __slots__ = ()

    def __repr__(self):
        """
        Return the string representation of the exception.
        """

        return str(self)


class WorkerException:
    """
    Worker exception.
    """

    def __init__(self, worker_id=None, exc_info=None):
        self.worker_id = worker_id
        self.pid = os.getpid()
        exc_info = exc_info if exc_info is not None else sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def reraise(self):
        """
        Reraise the exception.
        """

        if self.worker_id is not None:
            process_msg = f"DataLoader worker {self.worker_id}"
        else:
            process_msg = "DataLoader main process"
        exc_msg = process_msg + f" (pid: {self.pid}) caught {self.exc_type.__name__} with message:\n{self.exc_msg}"
        if self.exc_type is KeyError:
            exc_msg = KeyErrorMsg(exc_msg)
        try:
            raise self.exc_type(message=exc_msg)
        except Exception:
            raise RuntimeError(exc_msg) from None


class ParentProcessMonitor:
    """
    Parent process monitor.
    """

    def __init__(self):
        self.ppid = os.getppid()

    def is_alive(self):
        """
        Check if the parent process is alive.
        """

        return os.getppid() == self.ppid


def data_worker_fn(
    dataset, fetcher, num_workers, worker_id, index_queue, data_queue, worker_done, worker_init_fn, base_seed
):
    """
    Data worker function.
    """

    try:
        try:
            parent_process_monitor = ParentProcessMonitor()
            cde.register_worker_handlers()

            worker_seed = base_seed + worker_id
            random.seed(worker_seed)
            mindspore.set_seed(worker_seed & 0xFFFFFFFF)  # set seed for mindspore.ops and mindspore.dataset
            mindspore.manual_seed(worker_seed)  # set seed for mindspore.mint
            np.random.seed(seed_sequence(base_seed, worker_id))

            global WORKER_INFO_LOCAL
            WORKER_INFO_LOCAL = WorkerInfo(id=worker_id, num_workers=num_workers, seed=worker_seed, dataset=dataset)

            if worker_init_fn is not None:
                worker_init_fn(worker_id)
            fetcher.reset()
        except Exception:  # pylint: disable=W0703
            exc = WorkerException(worker_id)
            data_queue.put((-1, exc))
            return

        iteration_finished = False
        while parent_process_monitor.is_alive():
            try:
                index_item = index_queue.get(timeout=WORKER_TIME_OUT)
            except Empty:
                continue
            if isinstance(index_item, ResumeIterationFlag):
                data_queue.put((index_item, None))
                iteration_finished = False
                fetcher.reset()
                continue
            if index_item is None:
                if not worker_done.is_set() and not iteration_finished:
                    raise RuntimeError(
                        f"DataLoader worker {worker_id} (pid: {os.getpid()}) got None from index queue "
                        f"before quit flag is set."
                    )
                break  # we got the last data of index queue, now can safely quit
            if worker_done.is_set() or iteration_finished:
                # main process send quit flag, but we still need to empty the index queue, skip get data from dataset
                continue
            order_index, data_index = index_item
            try:
                data = fetcher.fetch(data_index)
            except StopIteration:
                iteration_finished = True
                # use None as a flag to tell the main process the iteration is finished
                data_queue.put((order_index, None))
                # continue here to wait for the main process to quit the worker
                continue
            except Exception:  # pylint: disable=W0703
                data = WorkerException(worker_id)
            data_queue.put((order_index, data))
            del order_index, data_index, data, index_item

    except KeyboardInterrupt:
        logger.info(f"DataLoader worker {worker_id} (pid: {os.getpid()}) was interrupted by the keyboard.")
    if worker_done.is_set():
        data_queue.close()
        data_queue.join_thread()


def get_worker_info():
    """
    Get the information about the current :class:`~mindspore.dataset.dataloader.DataLoader` worker process.

    The information includes:

    - id (:py:class:`int`): The ID of the current worker process.
    - num_workers (:py:class:`int`): The total number of the worker processes.
    - seed (:py:class:`int`): The random seed used by the current worker process. This value is determined
      by the base seed generated by the main process and the ID of the current worker process.
    - dataset (:class:`~mindspore.dataset.dataloader.Dataset`): The dataset object copied from the main
      process to the current worker process.

    If the current process is not a :class:`~mindspore.dataset.dataloader.DataLoader` worker process, return ``None``.

    Returns:
        Union[WorkerInfo, None], the information about the current :class:`~mindspore.dataset.dataloader.DataLoader`
        worker process.

    Examples:
        >>> from mindspore.dataset.dataloader import DataLoader, IterableDataset, get_worker_info
        >>>
        >>> # Split workload according to the worker info in multi-process data loading
        >>> class IterableStyleDataset(IterableDataset):
        ...     def __init__(self, num_samples):
        ...         self.start = 0
        ...         self.end = num_samples
        ...
        ...     def __iter__(self):
        ...         worker_info = get_worker_info()
        ...         if worker_info is None:  # single-process data loading
        ...             return iter(range(self.start, self.end))
        ...         else:  # multi-process data loading
        ...             return iter(range(worker_info.id, self.end, worker_info.num_workers))
        >>>
        >>> dataset = IterableStyleDataset(2)
        >>> dataloader = DataLoader(dataset, num_workers=2)
        >>> print(list(dataloader))
        [Tensor(shape=[1], dtype=Int64, value= [0]), Tensor(shape=[1], dtype=Int64, value= [1])]
    """
    return WORKER_INFO_LOCAL
