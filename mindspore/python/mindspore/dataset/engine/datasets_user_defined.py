# Copyright 2019-2025 Huawei Technologies Co., Ltd
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
import atexit
import builtins
import copy
import errno
import itertools
import math
import multiprocessing
import os
import platform
import queue
import signal
import subprocess
import threading
import time
import weakref
from functools import partial
from types import GeneratorType

import dill
import numpy as np
import psutil

import mindspore._c_dataengine as cde
from mindspore import log as logger
from mindspore.common import Tensor
from mindspore.communication.management import get_group_size
from . import samplers
from .datasets import UnionBaseDataset, MappableDataset, Schema, to_list, _PythonMultiprocessing
from .queue import _SharedQueue
from .validators import check_generator_dataset, check_numpy_slices_dataset, check_padded_dataset
from ..core.config import _get_enable_shared_mem, get_prefetch_size, get_multiprocessing_timeout_interval, \
    get_enable_watchdog, get_debug_mode, get_seed, set_seed, get_multiprocessing_start_method, get_video_backend, \
    set_video_backend
from ..core.datatypes import mstypelist_to_detypelist
from ..core.py_util_helpers import ExceptionHandler
from ..core.validator_helpers import type_check
from ..transforms import transforms


def _check_shm_usage(num_worker, queue_size, in_rowsize, out_rowsize):
    """
    Check sufficient shared memory is available for shared memory queues
    when training in parallel mode.
    """
    threshold_ratio = 0.8
    # Verify available size only when using static shared memory on Linux
    if platform.system().lower() not in {"windows", "darwin"} and in_rowsize != -1 and out_rowsize != -1:
        device_num = get_group_size()
        # In the cluster, _get_device_num indicates the number of the entire cluster. The maximum number of cards
        # on the ascend server is 8.
        if device_num > 1:
            device_num = min(device_num, 8)
        shm_estimate_usage = device_num * num_worker * \
                             (queue_size + 2) * (in_rowsize + out_rowsize) * 1024 * 1024
        try:
            shm_available = psutil.disk_usage('/dev/shm').free
            if shm_estimate_usage >= threshold_ratio * shm_available:
                raise RuntimeError(
                    f"Insufficient shared memory available. Required: {shm_estimate_usage}, " +
                    f"Available: {shm_available}. The required memory can't exceed 80% of the available " +
                    "shared memory, it's recommended to reduce memory usage by following methods:\n" +
                    "1. reduce value of parameter max_rowsize or num_parallel_workers.\n" +
                    "2. reduce prefetch size by set_prefetch_size().\n" +
                    "3. disable shared memory by set_enable_shared_mem().")
        except FileNotFoundError as exc:
            raise RuntimeError("Expected /dev/shm to exist.") from exc


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


def _cpp_sampler_fn(dataset, sample_ids):
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


def _cpp_sampler_fn_mp(sample_fn, sample_ids):
    """
    Multiprocessing generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    return sample_fn.process(sample_ids)


def _generator_fn_wrapper(function, *args):
    """
    Generate a new function that wraps the specified generator function with partial
    application of the given arguments and keywords.
    """
    return partial(function, *args)


def _fill_worker_indices(workers, indices, idx_cursor, worker_to_quit):
    """
    Worker index queue filler, fill worker index queue in round robin order or QUIT flag.
    """
    num_worker = len(workers)
    if idx_cursor < len(indices):
        while idx_cursor < len(indices):
            try:
                workers[idx_cursor % num_worker].put(indices[idx_cursor])
                idx_cursor += 1
            except queue.Full:
                break
    else:
        for i in range(num_worker):
            # just put only one QUIT flag to the sub-thread / sub-process
            if str(i) not in worker_to_quit:
                try:
                    workers[i].put("QUIT")
                    worker_to_quit[str(i)] = "QUIT"
                except queue.Full:
                    continue
    return idx_cursor, worker_to_quit


def _convert_row(row):
    """
    Convert Op return value to numpy, or keep as a dict (if already a dict)
    """

    # convert single item to np.array
    prim_type = (int, float, str, bytes, np.ndarray, Tensor, np.number, np.bool_)
    if isinstance(row, prim_type):
        if isinstance(row, Tensor):  # mindspore.Tensor
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
        if isinstance(x, Tensor):  # mindspore.Tensor
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


class SamplerFn(cde.PythonMultiprocessingRuntime):
    """
    Multiprocessing or multithread generator function wrapper master process.
    """

    def __init__(self, dataset, num_worker, multi_process, max_rowsize):
        super().__init__()
        self.workers = []
        self.dataset = dataset
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.max_rowsize = max_rowsize
        self.need_join = False

    def is_mp_enabled(self):
        return self.workers is not None and self.workers

    def launch(self, op_id=-1):
        """launch the multiprocessing pool"""
        self.op_id = op_id
        logger.info("Launching new Python Multiprocessing pool for GeneratorOp:" + str(self.op_id))
        if self.is_mp_enabled():
            message = "Launching a new Python multiprocessing pool for GeneratorOp while a pool already exists!" + \
                " The existing pool will be terminated first."
            logger.warning(message)
            self._stop_subprocess()
            self.reset()
            self.workers = []

        self.ppid = os.getpid()
        self.pids = []
        self.thread_ids = []
        self.check_interval = get_multiprocessing_timeout_interval()  # the interval of check queue's size

        if self.multi_process is True:
            multiprocessing.set_start_method(get_multiprocessing_start_method(), True)
            # Event for end of epoch
            try:
                self.eof = multiprocessing.Event()
            except Exception as exc:
                raise RuntimeError("Init multiprocessing.Event() failed, This might be caused by insufficient shm,"
                                   + " and the recommended shm size is at least 5 GB.") from exc

            # Create workers
            # get default queue size and adjust queue size per worker if there are large # workers
            queue_size = get_prefetch_size()
            queue_size = min(queue_size, queue_size * 4 // self.num_worker)
            queue_size = max(2, queue_size)

            if _get_enable_shared_mem():
                # generator dataset use idx_queue and res_queue to transfer data between main and subprocess
                # idx_queue is used multiprocess.Queue which is not shared memory, so it's size is 0.
                # res_queue is used shared memory, so its size is max_rowsize which is defined by user.
                _check_shm_usage(self.num_worker, queue_size, 0, self.max_rowsize)
            self.count = multiprocessing.Value('i', 0)
            for worker_id in range(self.num_worker):
                try:
                    logger.info(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
                    worker = _GeneratorWorkerMp(self.dataset, self.eof, self.max_rowsize, queue_size, self.ppid,
                                                self.count, worker_id)
                    worker.daemon = True
                    # When multi processes fork a subprocess, the lock of the main process is copied to the subprocess,
                    # which may cause deadlock. Therefore, the subprocess startup is performed in the initialization
                    # phase. In this phase, the main process is not locked.
                    worker.start()
                except OSError as exc:
                    if exc.errno == errno.EMFILE:
                        raise RuntimeError("Failed to launch multiprocessing of GeneratorDataset: "
                                           "Too many open files. Please check if `num_parallel_workers` "
                                           "is set too large, or you are creating iterators multiple times. "
                                           "You can also increase the limit using `ulimit -n` in the shell "
                                           "to avoid this error.") from exc
                    raise
                except Exception as exc:
                    raise RuntimeError("Failed to launch multiprocessing of GeneratorDataset.") from exc
                self.pids.append(worker.pid)
                self.need_join = True
                self.workers.append(worker)
            multiprocessing.set_start_method("fork", True)

            logger.info(f"Launch generator worker process(es): {[worker.pid for worker in self.workers]}")
            if platform.system().lower() != 'windows':
                self._launch_monitor()
        else:
            self.eof = threading.Event()
            for worker_id in range(self.num_worker):
                worker = _GeneratorWorkerMt(self.dataset, self.eof, worker_id)
                worker.daemon = True
                self.need_join = True
                worker.start()
                self.thread_ids.append(worker.ident)
                self.workers.append(worker)

        # Register a termination function using weakref to avoid the object from unable to properly destruct.
        atexit.register(lambda cleanup: cleanup()() if cleanup() is not None else None,
                        weakref.WeakMethod(self.terminate))

    def get_worker_ids(self):
        """
        Get dict of worker's ids

        Returns:
            dict of strings
        """
        if not self.is_mp_enabled():
            return {}
        worker_ids = {}
        if self.multi_process is True:
            worker_ids["is_thread"] = False
            worker_ids["worker_id"] = self.pids
        else:
            worker_ids["is_thread"] = True
            worker_ids["worker_id"] = self.thread_ids
        return worker_ids

    def terminate(self):
        self._stop_subprocess()

    def _interval_log(self, i, start_time, wait_count):
        cost_time = int(time.time()) - start_time
        if cost_time / self.check_interval >= wait_count:
            wait_count += 1
            self._log_stuck_warning(self.workers[i % self.num_worker], cost_time)
        return wait_count

    def _check_and_start_process(self):
        """Check the idx_queue and start the process"""
        if self.workers is None:
            raise RuntimeError("The GeneratorDataset subprocess worker may be killed or exit abnormally.")
        for w in self.workers:
            # Check whether the queue of the subprocess is empty.
            if not w.queue_empty():
                # in failover reset scenario the QUIT flag should be pop first
                while w.idx_queue.qsize() > 0:
                    try:
                        result = w.idx_queue.get(timeout=1)
                        if result != "QUIT":
                            raise Exception("The queue of the subprocess is not empty.")
                    except queue.Empty:
                        continue
            # Start all workers
            if not w.is_alive():
                try:
                    w.start()
                except RuntimeError as e:
                    # the worker may be being started.
                    if w._started.is_set():  # pylint: disable=W0212
                        continue
                    raise e

    def process(self, indices):
        """
        The main process, start the child process or child thread, and fill the index queue.
        Get the result and return.
        """
        self._check_and_start_process()

        # Fill initial index queues
        idx_cursor = 0
        # worker to quit
        worker_to_quit = {}
        idx_cursor, worker_to_quit = _fill_worker_indices(self.workers, indices, idx_cursor, worker_to_quit)

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
                    time.sleep(0.1)
                    if self.eof.is_set():
                        logger.warning("Generator receives a termination signal, stop waiting for data "
                                       "from subprocess.")
                        self._stop_subprocess()
                        return
                    wait_count = self._interval_log(i, start_time, wait_count)
                result = self.workers[i % self.num_worker].get()
                if isinstance(result, ExceptionHandler):
                    result.reraise()
            except queue.Empty as exc:
                self._stop_subprocess()
                raise RuntimeError("Generator worker process timeout.") from exc
            except KeyboardInterrupt as exc:
                self._stop_subprocess()
                raise RuntimeError("Generator worker receives KeyboardInterrupt.") from exc
            if self.eof.is_set():
                self._stop_subprocess()
                return

            idx_cursor, worker_to_quit = _fill_worker_indices(self.workers, indices, idx_cursor, worker_to_quit)

            yield _convert_row(result)

    def _log_stuck_warning(self, worker, waiting_time):
        """
        Log warning of the stuck worker, containing the worker ID, waiting time and
        the current stack (if py-spy installed).

        Args:
            worker (Union[threading.Thread, multiprocessing.Process]): The worker instance.
            waiting_time (int): The waiting time for getting data from the worker.
        """
        if self.multi_process:
            stuck_worker_id = worker.pid
            worker_type = "process"
            stuck_pid = stuck_worker_id
        else:
            if hasattr(worker, "native_id"):
                # only supported since Python 3.8
                stuck_worker_id = worker.native_id
            else:
                stuck_worker_id = worker.ident
            worker_type = "thread"
            stuck_pid = os.getpid()  # get the process ID of the stuck thread
        warning_message = "Has been waiting for data from Generator worker {0} ID '{1}' " \
                          "for more than {2} seconds. Please check if the user defined " \
                          "dataset of GeneratorDataset has a dead loop, or is processing " \
                          "too slowly. ".format(worker_type, stuck_worker_id, waiting_time)
        install_status, _ = subprocess.getstatusoutput("py-spy --version")
        if install_status == 0:
            stack = subprocess.getoutput("py-spy dump -p {}".format(stuck_pid))
            warning_message += "Below is the stack of this worker:\n{0}\n".format(stack)
        else:
            warning_message += "You can install py-spy via `pip install py-spy`, then " \
                               "stop and rerun your script to get the current stack. "
        warning_message += "If it is not a problem, you can adjust the printing frequency of this log via " \
                           "the `mindspore.dataset.config.set_multiprocessing_timeout_interval` interface."
        logger.warning(warning_message)

    def _launch_monitor(self):
        """
        Launch a clean process and register subprocess to be monitored by the watch dog.
        The clean process will clean up subprocesses when main process exited.
        The watch dog will clean up subprocesses and main process when any subprocess exited.
        """
        _clean_worker_func = _PythonMultiprocessing._clean_process  # pylint: disable=W0212
        self.cleaning_process = multiprocessing.Process(target=_clean_worker_func,
                                                        name="GeneratorCleanProcess",
                                                        args=(self.ppid, self.workers, self.eof))
        self.cleaning_process.daemon = True
        self.cleaning_process.start()
        logger.info("Launch clean process {} to monitor worker "
                    "process(es): {}".format(self.cleaning_process.pid, [worker.pid for worker in self.workers]))

        if get_enable_watchdog():
            worker_ids = [os.getpid()]
            worker_ids.extend([worker.pid for worker in self.workers])
            worker_ids.append(self.cleaning_process.pid)
            cde.register_worker_pids(id(self), worker_ids)

    def _release_fd(self):
        """Release the file descriptor by subprocess"""
        # release the file descriptor handle
        check_interval = get_multiprocessing_timeout_interval()
        for w in self.workers:
            try:
                subprocess_file_descriptor = w.sentinel
                st = time.time()
                while _PythonMultiprocessing.is_process_alive(w.pid):
                    process = psutil.Process(w.pid)
                    if process.status() == psutil.STATUS_ZOMBIE:
                        process.kill()
                        break
                    time.sleep(0.01)  # sleep 10ms, waiting for the subprocess exit
                    if time.time() - st > check_interval:
                        logger.warning("Waiting for the subprocess worker [{}] to exit.".format(w.pid))
                        st += check_interval
            except ValueError as e:
                if "process object is closed" in str(e):
                    continue
                raise e
            try:
                if w.is_alive():
                    os.close(subprocess_file_descriptor)
            except OSError as e:
                # Maybe the file descriptor had been released, so ignore the 'Bad file descriptor'
                if "Bad file descriptor" not in str(e):
                    raise e
            except AttributeError:  # maybe occur "'NoneType' object has no attribute 'maxsize'"
                pass

    def _stop_subprocess(self):
        """Only the main process can call join. All the sub-process / sub-thread will be stopped."""
        if self.need_join is True and self.ppid == os.getpid():
            self.need_join = False
            # abort the monitor first
            self._abort_monitor()

            # waiting for the sub-process stop
            for w in self.workers:
                if self.multi_process is True and hasattr(w, '_closed') and w._closed is False:  # pylint: disable=W0212
                    try:
                        # del the queue first
                        del w.res_queue
                        del w.idx_queue

                        # let the quit event notify the worker process to exit
                        w.join(timeout=5)
                        if _PythonMultiprocessing.is_process_alive(w.pid):
                            # if the worker process did not exit, it may hang, try to terminate it
                            w.terminate()
                            w.close()
                    except Exception:  # pylint: disable=W0703
                        # Block all errors when join
                        continue
                elif not self.multi_process:
                    w.join(timeout=5)

            if self.multi_process is True:
                self._release_fd()

            self.workers.clear()
            self.workers = None
            # Under independent processes, the GeneratorDataset pulls up multiple processes in a spawn manner, and
            # after the use case exits normally, there will be a warning: UserWarning: resource_tracker: There appear
            # to be %d leaked semaphore objects to clean up at shutdown.
            self.eof = None

    def _abort_monitor(self):
        """Deregister workers monitored by the watch dog and join clean process."""
        if get_enable_watchdog():
            cde.deregister_worker_pids(id(self))
        if hasattr(self, 'eof') and self.eof is not None:
            self.eof.set()
            # send QUIT flag to workers, and the worker's while loop could check the eof flag
            for worker in self.workers:
                if not worker.queue_full():
                    worker.put("QUIT")
        if hasattr(self, 'cleaning_process') and self.cleaning_process is not None:
            # let the quit event notify the cleaning process to exit
            self.cleaning_process.join(timeout=5)
            if self.cleaning_process.is_alive():
                # if the cleaning process did not exit, it may hang, try to terminate it
                _PythonMultiprocessing._terminate_processes([self.cleaning_process])  # pylint: disable=W0212
            del self.cleaning_process
        if hasattr(self, 'count'):
            del self.count

    def __del__(self):
        try:
            self._stop_subprocess()
        except TypeError:
            pass

    def __deepcopy__(self, memodict, exclude=()):
        self.__init__(self.dataset, self.num_worker, self.multi_process, self.max_rowsize)


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


def _generator_worker_loop(dataset, idx_queue, result_queue, eof, is_multiprocessing, worker_id, ppid=-1,
                           video_backend=None):
    """
    Multithread or multiprocess generator worker process loop.
    """
    # Initialize C++ side signal handlers
    cde.register_worker_handlers()

    if is_multiprocessing:
        if video_backend is not None:
            set_video_backend(video_backend)

        result_queue.cancel_join_thread()  # Ensure that the process does not hang when exiting

        # init the random seed and np.random seed for the subprocess
        if get_seed() != 5489:
            set_seed(get_seed() + worker_id)

    while not eof.is_set():
        _ignore_sigint(is_multiprocessing=is_multiprocessing)

        # Fetch index, block
        try:
            idx = idx_queue.get(timeout=1)
        except queue.Empty:
            if _main_process_already_exit(eof, is_multiprocessing, idx_queue, result_queue, ppid) is True:
                del idx_queue
                del result_queue
                return
            # If end-of-file (eof) is not set, continue to get data from idx_queue
            continue
        if idx == "QUIT":
            # all the data had been processed, so we release the executor which is used by the current thread/process
            transforms.clean_unused_executors()
            continue
        if idx is None:
            # When the queue is out of scope from master process, a None item can be fetched from the queue.
            # Upon receiving None, worker process should check if eof is set.
            if not eof.is_set():
                raise Exception("")
            del idx_queue
            del result_queue
            return
        if eof.is_set():
            del idx_queue
            del result_queue
            return
        # Fetch data, any exception from __getitem__ will terminate worker and timeout master process
        try:
            result = dataset[idx]
        except Exception:  # pylint: disable=broad-except
            result = ExceptionHandler(where="in GeneratorDataset worker process")
        # Send data, block
        while not eof.is_set():
            try:
                result_queue.put(result, timeout=5)
            except queue.Full:
                if _main_process_already_exit(eof, is_multiprocessing, idx_queue, result_queue, ppid) is True:
                    del idx_queue
                    del result_queue
                    return
                # If eof is not set, continue to put data to result_queue
                continue
            break
        del result, idx


class _GeneratorWorkerMt(threading.Thread):
    """
    Worker process for multi-thread Generator.
    """

    def __init__(self, dataset, eof, worker_id):
        self.idx_queue = queue.Queue(16)
        self.res_queue = queue.Queue(16)
        super().__init__(target=_generator_worker_loop,
                         args=(dataset, self.idx_queue, self.res_queue, eof, False, worker_id),
                         name="GeneratorWorkerThread" + str(worker_id))

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

    def queue_full(self):
        return self.idx_queue.full()


class _GeneratorWorkerMp(multiprocessing.Process):
    """
    Worker process for multiprocess Generator.
    """

    def __init__(self, dataset, eof, max_rowsize, queue_size, ppid, count, worker_id):
        self.idx_queue = multiprocessing.Queue(queue_size)
        if _get_enable_shared_mem():
            self.res_queue = _SharedQueue(queue_size, count, max_rowsize=max_rowsize)
        else:
            self.res_queue = multiprocessing.Queue(queue_size)
        self.idx_queue.cancel_join_thread()  # Ensure that the process does not hang when exiting
        video_backend = get_video_backend() if multiprocessing.get_start_method() == "spawn" else None
        super().__init__(target=_generator_worker_loop,
                         args=(dataset, self.idx_queue, self.res_queue, eof, True, worker_id, ppid, video_backend),
                         name="GeneratorWorkerProcess" + str(worker_id))

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

    def queue_full(self):
        return self.idx_queue.full()

    def __del__(self):
        # del all the Queue & SharedQueue when the iter had been deleted from ITERATORS_LIST
        if hasattr(self, 'idx_queue'):
            del self.idx_queue
        if hasattr(self, 'res_queue'):
            # del the queue when has
            del self.res_queue


class _GeneratorWrapper:
    """Wrapper the generator so that it can be iterated multiple times in GeneratorDataset."""

    def __init__(self, generator):
        self.generator = generator
        self.generator_new, self.generator = itertools.tee(self.generator)

    def __iter__(self):
        self.generator_new, self.generator = itertools.tee(self.generator)
        return self

    def __next__(self):
        return next(self.generator_new)


class _PickleGeneratorSource:
    """Starting multiple processes in spawn mode requires pickling source object in GeneratorDataset."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        state = dill.dumps(self.dataset)
        return state

    def __setstate__(self, state):
        self.dataset = dill.loads(state)


class GeneratorDataset(MappableDataset, UnionBaseDataset):
    """
    A source dataset that generates data from Python by invoking Python data source each epoch.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        source (Union[Random Accessible, Iterable]): A custom dataset from which to load the data.
            MindSpore supports the following types of datasets:

            - Random-accessible (map-style) datasets: A dataset object that implements the `__getitem__()`
              and `__len__()` methods, represents a mapping from indexes/keys to data samples.
              For example, such a dataset `source`, when accessed with `source[idx]`, can read the idx-th sample
              from disk, see `Random-accessible dataset example <https://www.mindspore.cn/tutorials/en/master/
              beginner/dataset.html#random-accessible-dataset>`_ for details.

            - Iterable-style dataset: An iterable dataset object that implements `__iter__()` and `__next__()` methods,
              represents an iterable over data samples. This type of dataset is suitable for situations where
              random reads are costly or even impossible, and where batch sizes depend on the data being acquired.
              For example, such a dataset `source`, when accessed `iter(source)`, can return a stream of data reading
              from a database or remote server, see `Iterable-style dataset example
              <https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#iterable-dataset>`_ for details.

        column_names (Union[str, list[str]], optional): List of column names of the dataset. Default: ``None`` .
            Users are required to provide either column_names or schema.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset. Default: ``None`` .
            If provided, sanity check will be performed on generator output (deprecated in future version).
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`mindspore.dataset.Schema` are
            acceptable (deprecated in future version). Default: ``None`` .
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None`` , all images.
        num_parallel_workers (int, optional): Number of worker threads/subprocesses used to
            fetch the dataset in parallel. Default: ``1``.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            Default: ``None`` , expected order behavior shown in the table below.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required. Default: ``None`` , expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            Random accessible input is required. When this argument is specified, `num_samples` reflects the maximum
            sample number of per shard. Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` .
            This argument must be specified only when `num_shards` is also specified.
            Random accessible input is required.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: ``True``.
        max_rowsize (int, optional): Maximum size of data (in MB) that is used for shared memory
            allocation to copy data between processes, the total occupied shared memory will increase as
            ``num_parallel_workers`` and :func:`mindspore.dataset.config.set_prefetch_size` increase. If set to ``-1``,
            shared memory will be dynamically allocated with the actual size of data. This is only used if
            ``python_multiprocessing`` is set to ``True``. Default: ``None`` , allocate shared memory dynamically
            (deprecated in future version).
        batch_sampler (Iterable, optional): Similar to `sampler` , but returns a batch of indices at a time, the
            corresponding data will be combined into a batch. Mutually exclusive with `num_samples` , `shuffle` ,
            `num_shards` , `shard_id` and `sampler` . Default: ``None`` , do not use batch sampler.
        collate_fn (Callable[List[numpy.ndarray]], optional): Define how to merge a list of data into a batch.
            Only valid if `batch_sampler` is used. Default: ``None`` , do not use collation function.

    .. warning::
        `GeneratorDataset` uses `dill` module implicitly in multiprocessing `spawn` mode to serialize/deserialize
        `source`, which is known to be insecure. It is possible to construct malicious pickle data which will
        execute arbitrary code during unpickling. Never load data that could have come from untrusted sources,
        or has been tampered with.

    Note:
        - The parameter `column_types` , `schema` and `max_rowsize` will be deprecated in a future version.
        - If you configure `python_multiprocessing=True` (Default: ``True`` ) and `num_parallel_workers>1`
          (default: ``1`` ) indicates that the multiprocessing mode is started for data load acceleration.
          At this time, as the dataset iterates, the memory consumption of the subprocess will gradually increase,
          mainly because the subprocess of the user-defined dataset obtains the member variables from the main
          process in the Copy On Write way.
          Example: If you define a dataset with `__init__` function which contains a large number of member variable
          data (for example, a very large file name list is loaded during the dataset construction) and uses the
          multiprocessing mode, which may cause the problem of OOM (the estimated total memory usage is:
          `(num_parallel_workers+1) * size of the parent process` ). The simplest solution is to replace Python objects
          (such as list/dict/int/float/string) with non referenced data types
          (such as Pandas, Numpy or PyArrow objects) for member variables, or load less metadata in member variables,
          or configure `python_multiprocessing=False` to use multi-threading mode.

          You can use the following classes/functions to reduce the size of member variables:

          :class:`mindspore.dataset.utils.LineReader`: Use this class to initialize your text file object in the
          `__init__` function. Then read the file content based on the line number of the object with the `__getitem__`
          function.

        - Input `source` accepts user-defined Python functions (PyFuncs), and sets the multiprocessing start method
          to `spawn` mode by ds.config.set_multiprocessing_start_method("spawn") with `python_multiprocessing=True`
          and `num_parallel_workers>1` supports adding network computing operators from mindspore.nn and mindspore.ops
          or others into this `source`, otherwise adding to the `source` is not supported.
        - When the user defined dataset by `source` calls the DVPP operator during dataset loading and processing,
          the supported scenarios are as follows:

          +---------------+----------------------------+----------------------------+----------------------------+
          |               |                            |                     Multiprocessing                     |
          |               |       Multithreading       +----------------------------+----------------------------+
          |               |                            |           spawn            |            fork            |
          +===============+============================+============================+============================+
          |Independent    |Data Processing: support    |Data Processing: support    |Data Processing: support    |
          |               |                            |                            |                            |
          |process mode   |Data Processing + Network   |Data Processing + Network   |Data Processing + Network   |
          |               |training: not support       |training: support           |training: not support       |
          +---------------+----------------------------+----------------------------+----------------------------+
          |Non-independent|Data Processing: support    |Data Processing: support    |Data Processing: support    |
          |               |                            |                            |                            |
          |process mode   |Data Processing + Network   |Data Processing + Network   |Data Processing + Network   |
          |               |training: support           |training: support           |training: not support       |
          +---------------+----------------------------+----------------------------+----------------------------+

        - The parameters `num_samples` , `shuffle` , `num_shards` , `shard_id` can be used to control the sampler
          used in the dataset, and their effects when combined with parameter `sampler` are as follows.

    .. include:: mindspore.dataset.sampler.txt

    Raises:
        RuntimeError: If source raises an exception during execution.
        RuntimeError: If len of column_names does not match output len of source.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If sampler and shuffle are specified at the same time.
        ValueError: If sampler and sharding are specified at the same time.
        ValueError: If `num_shards` is specified but shard_id is None.
        ValueError: If shard_id is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `batch_sampler` is specified together with `num_samples` ,
            `shuffle` , `num_shards` , `shard_id` and `sampler`.
        ValueError: If `collate_fn` is specified while `batch_sampler` is None.
        TypeError: If `batch_sampler` is not iterable.
        TypeError: If `collate_fn` is not callable.

    Examples:
        >>> import mindspore.dataset as ds
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

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_
    """

    @check_generator_dataset
    def __init__(self, source, column_names=None, column_types=None, schema=None, num_samples=None,
                 num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
                 python_multiprocessing=True, max_rowsize=None, batch_sampler=None, collate_fn=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
        if isinstance(source, builtins.zip):
            # Although zip is iterable, it does not have the feature of repeated iteration, so pass it to the array.
            self.source = list(source)
        else:
            self.source = source

        # wrapper the generator so that it can be iterated multiple times
        if isinstance(self.source, GeneratorType):
            self.source = _GeneratorWrapper(self.source)

        self.prepared_source = None  # source to be sent to C++
        self._check_operator_mixed()
        self._check_windows(self.num_parallel_workers, python_multiprocessing)

        if self.python_multiprocessing and get_debug_mode():
            logger.warning("Python multiprocessing is not supported in debug mode."
                           " Ignoring Python multiprocessing for GeneratorDataset.")

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

        self.has_batch_sampler = False
        if batch_sampler is not None:
            self.has_batch_sampler = True
            if not isinstance(batch_sampler, samplers.BuiltinSampler):
                self.sampler = samplers.IterSampler(batch_sampler)
            else:
                self.sampler = batch_sampler

        # Move get dataset_size by len from parse to here, because self.source will
        # lose attribution of '__len__' after deepcopy.
        self.source_len = len(self.source) if hasattr(self.source, "__len__") else -1

        self.max_rowsize = max_rowsize if max_rowsize is not None else -1
        self.sample_fn = None
        # Ignore batch_info in the input parameter.
        self.collate_fn = (lambda *args: collate_fn(*args[:-1])) if collate_fn is not None else None

    def _check_operator_mixed(self):
        """check whether operator mixed"""
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True and \
           get_multiprocessing_start_method() == "fork":
            self.num_parallel_workers = 1
            logger.warning(
                "Input 'source' of 'GeneratorDataset' includes network computing operators like in mindspore.nn, "
                "mindspore.ops, mindspore.numpy module and etc, which do not support multi-thread compiling, recommend"
                " to replace it with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' "
                "into 1.")

    def _check_windows(self, num_parallel_workers, python_multiprocessing):
        """disable multiprocess when windows"""
        if platform.system().lower() == 'windows' and num_parallel_workers > 1 and python_multiprocessing:
            logger.warning("Python multiprocessing is not supported on Windows platform.")
        self.python_multiprocessing = python_multiprocessing if platform.system().lower() != 'windows' else False

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        return self.__safe_deepcopy__(memodict, exclude=("source", "__transfer_dataset__"))

    def __getitem__(self, index):
        type_check(index, (int, np.number), "index")
        if not hasattr(self.source, "__getitem__"):
            raise RuntimeError("Dataset don't support randomized access.")
        if self.has_batch_sampler:
            raise RuntimeError("GeneratorDataset with batch_sampler does not support random access.")
        if not hasattr(self, "generator_op"):
            dataset = copy.deepcopy(self)
            self.prepared_source = _generator_fn_wrapper(_cpp_sampler_fn, self.source)
            if self.schema is None:
                dataset.generator_node = cde.GeneratorNode(self.prepared_source, self.column_names, self.column_types,
                                                           self.source_len, self.sampler, 1, None, False)
            else:
                schema = self.schema
                if isinstance(schema, Schema):
                    schema = self.schema.cpp_schema
                dataset.generator_node = cde.GeneratorNode(self.prepared_source, schema, self.source_len,
                                                           self.sampler, 1, None, False)
            self.generator_op = dataset.generator_node.Build()
        sample_id = self.generator_op.GetMappedIndex(index)
        return self.source[sample_id]

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
            if not self.has_batch_sampler:
                # If the source has __getitem__ attribute, call the split method of MappableDataset.
                # Otherwise, call the split method of Dataset.
                return super().split(sizes, randomize)
            logger.warning("The performance of split will be degraded since batch_sampler is detected.")
        return super(MappableDataset, self).split(sizes, randomize)

    def prepare_multiprocessing(self):
        """Preprocessing of prepared_source."""
        sample_fn = None
        if self.sampler is not None and hasattr(self.source, "__getitem__"):
            # The reason why there is a try catch here is because when the new op is being constructed with shared
            # memory enabled, there will be an exception thrown if there is not enough shared memory available
            if self.source_len == -1:
                raise RuntimeError("Attempt to construct a random access dataset, '__len__' method is required!")

            if self.num_parallel_workers > 1 and not get_debug_mode():
                self.__validate_memory_usage()
                # Starting multiple processes in spawn mode requires pickling source object
                self.source = _PickleGeneratorSource(self.source)

                sample_fn = SamplerFn(self.source, self.num_parallel_workers, self.python_multiprocessing,
                                      self.max_rowsize)
                self.prepared_source = _generator_fn_wrapper(_cpp_sampler_fn_mp, sample_fn)
            else:
                self.prepared_source = _generator_fn_wrapper(_cpp_sampler_fn, self.source)
            self.sample_fn = sample_fn
        else:
            self.sampler = None
            self.sample_fn = sample_fn
            self.source_len = min(self.source_len, self.num_samples) if self.num_samples != 0 else self.source_len
            if not hasattr(self.source, "__iter__"):
                # Use generator function if input callable
                self.prepared_source = _generator_fn_wrapper(_generator_fn, self.source, self.num_samples)
            else:
                # Use iterator function if input is iterable
                # Random accessible input is also iterable
                self.prepared_source = _generator_fn_wrapper(_iter_fn, self.source, self.num_samples)

    def parse(self, children=None):
        self.prepare_multiprocessing()
        if self.schema is None:
            return cde.GeneratorNode(self.prepared_source, self.column_names, self.column_types, self.source_len,
                                     self.sampler, self.num_parallel_workers, self.sample_fn, self.has_batch_sampler)
        schema = self.schema
        if isinstance(schema, Schema):
            schema = self.schema.cpp_schema
        return cde.GeneratorNode(self.prepared_source, schema, self.source_len, self.sampler,
                                 self.num_parallel_workers, self.sample_fn, self.has_batch_sampler)

    def __validate_memory_usage(self):
        """
        Check memory usage when multiprocessing mode, when 85% prompt warning and 100% raise error.
        """
        if self.python_multiprocessing:
            # setting num_parallel_workers too large when using python multiprocessing may cause
            # out of memory for getting num_shards
            valid_num_shards = 1
            if isinstance(self.sampler, samplers.DistributedSampler):
                valid_num_shards = self.sampler.num_shards
            elif self.num_shards is not None:
                valid_num_shards = self.num_shards

            # get process memory usage
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss
            sys_memory_available = psutil.virtual_memory().available

            total_memory_maybe_used = process_memory * self.num_parallel_workers * valid_num_shards
            if total_memory_maybe_used / sys_memory_available > 0.85:
                valid_num_worker = math.floor(sys_memory_available * 0.85 / valid_num_shards / process_memory)
                valid_num_worker = 1 if valid_num_worker <= 0 else valid_num_worker
                info = "GeneratorDataset's num_parallel_workers: {} is too large which may cause a lot of memory " \
                       "occupation (>85%) or out of memory(OOM) during multiprocessing. Therefore, it is recommended " \
                       "to reduce num_parallel_workers to {} or smaller.".format(self.num_parallel_workers,
                                                                                 valid_num_worker)
                logger.warning(info)

    def add_sampler(self, new_sampler):
        """
        Add a child sampler for the current dataset.

        Note:
            - If the sampler is added and it has a shuffle option, its value must be ``Shuffle.GLOBAL`` .
              Additionally, the original sampler's shuffle value cannot be ``Shuffle.PARTIAL`` .

        Args:
            new_sampler (Sampler): The child sampler to be added.

        Examples:
            >>> import mindspore.dataset as ds
            >>> dataset = ds.GeneratorDataset([i for i in range(10)], "column1")
            >>>
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.add_sampler(new_sampler)
        """
        # check PKSampler
        if isinstance(new_sampler, samplers.PKSampler):
            raise RuntimeError("GeneratorDataset doesn't support PKSampler")

        # call parent class
        super().add_sampler(new_sampler)


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
        data (Union[list, tuple, dict]): Input of given data. Supported data types include: list, tuple, dict and other
            NumPy formats. Input data will be sliced along the first dimension and generate additional rows, if input is
            list, there will be one column in each row, otherwise there tends to be multi columns. Large data is not
            recommended to be loaded in this way as data is loading into memory.
        column_names (list[str], optional): List of column names of the dataset. Default: ``None`` . If `column_names`
            is not provided, the output column names will be named as the keys of dict when the input data is a dict,
            otherwise they will be named like column_0, column_1 ...
        num_samples (int, optional): The number of samples to be included in the dataset. Default: ``None`` ,
            all samples.
        num_parallel_workers (int, optional): Number of worker subprocesses used to
            fetch the dataset in parallel. Default: ``1``.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset.
            Default: ``None`` , expected order behavior shown in the table below.
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset.
            Default: ``None`` , expected order behavior shown in the table below.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
            Used in `data parallel training <https://www.mindspore.cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This argument must be
            specified only when `num_shards` is also specified.

    Note:
        - The parameters `num_samples` , `shuffle` , `num_shards` , `shard_id` can be used to control the sampler
          used in the dataset, and their effects when combined with parameter `sampler` are as follows.

    .. include:: mindspore.dataset.sampler.txt

    Raises:
        RuntimeError: If len of column_names does not match output len of data.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If sampler and shuffle are specified at the same time.
        ValueError: If sampler and sharding are specified at the same time.
        ValueError: If `num_shards` is specified but shard_id is None.
        ValueError: If shard_id is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Examples:
        >>> import mindspore.dataset as ds
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

    @check_numpy_slices_dataset
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
        padded_samples (list[dict]): Samples provided by user.

    Raises:
        TypeError: If padded_samples is not an instance of list.
        TypeError: If the element of padded_samples is not an instance of dict.
        ValueError: If the padded_samples is empty.

    Tutorial Examples:
        - `Load & Process Data With Dataset Pipeline
          <https://www.mindspore.cn/docs/en/master/api_python/samples/dataset/dataset_gallery.html>`_

    Examples:
        >>> import mindspore.dataset as ds
        >>> import numpy as np
        >>> data = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)}]
        >>> dataset = ds.PaddedDataset(padded_samples=data)
    """

    @check_padded_dataset
    def __init__(self, padded_samples):
        dataset = _PaddedDataset(padded_samples)
        super().__init__(dataset, column_names=dataset.column_names, num_shards=None, shard_id=None, shuffle=False)
        self._dataset_size = len(dataset.padded_samples)
        self.padded_samples = padded_samples
