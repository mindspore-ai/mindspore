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
"""Schedule the event writer process."""
import multiprocessing as mp
from enum import Enum, unique
from mindspore import log as logger
from ..._c_expression import Tensor
from ._summary_adapter import SummaryType, package_summary_event, save_summary_data

# define the type of summary
FORMAT_SCALAR_STR = "Scalar"
FORMAT_TENSOR_STR = "Tensor"
FORMAT_IMAGE_STR = "Image"
FORMAT_HISTOGRAM_STR = "Histogram"
FORMAT_BEGIN_SLICE = "[:"
FORMAT_END_SLICE = "]"

# cache the summary data dict
# {id: SummaryData}
#           |---[{"name": tag_name, "data": numpy}, {"name": tag_name, "data": numpy},...]
g_summary_data_id = 0
g_summary_data_dict = {}
# cache the summary data file
g_summary_writer_id = 0
g_summary_file = {}


@unique
class ScheduleMethod(Enum):
    """Schedule method type."""
    FORMAL_WORKER = 0    # use the formal worker that receive small size data by queue
    TEMP_WORKER = 1      # use the Temp worker that receive big size data by the global value(avoid copy)
    CACHE_DATA = 2       # Cache data util have idle worker to process it


@unique
class WorkerStatus(Enum):
    """Worker status."""
    WORKER_INIT = 0        # data is exist but not process
    WORKER_PROCESSING = 1  # data is processing
    WORKER_PROCESSED = 2   # data already processed


def _parse_tag_format(tag: str):
    """
    Parse the tag.

    Args:
        tag (str): Format: xxx[:Scalar] xxx[:Image] xxx[:Tensor].

    Returns:
        Tuple, (SummaryType, summary_tag).
    """

    summary_type = SummaryType.INVALID
    summary_tag = tag
    if tag is None:
        logger.error("The tag is None")
        return summary_type, summary_tag

    # search the slice
    slice_begin = FORMAT_BEGIN_SLICE
    slice_end = FORMAT_END_SLICE
    index = tag.rfind(slice_begin)
    if index is -1:
        logger.error("The tag(%s) have not the key slice.", tag)
        return summary_type, summary_tag

    # slice the tag
    summary_tag = tag[:index]

    # check the slice end
    if tag[-1:] != slice_end:
        logger.error("The tag(%s) end format is error", tag)
        return summary_type, summary_tag

    # check the type
    type_str = tag[index + 2: -1]
    logger.debug("The summary_tag is = %r", summary_tag)
    logger.debug("The type_str value is = %r", type_str)
    if type_str == FORMAT_SCALAR_STR:
        summary_type = SummaryType.SCALAR
    elif type_str == FORMAT_TENSOR_STR:
        summary_type = SummaryType.TENSOR
    elif type_str == FORMAT_IMAGE_STR:
        summary_type = SummaryType.IMAGE
    elif type_str == FORMAT_HISTOGRAM_STR:
        summary_type = SummaryType.HISTOGRAM
    else:
        logger.error("The tag(%s) type is invalid.", tag)
        summary_type = SummaryType.INVALID

    return summary_type, summary_tag


class SummaryDataManager:
    """Manage the summary global data cache."""
    def __init__(self):
        global g_summary_data_dict
        self.size = len(g_summary_data_dict)

    @classmethod
    def summary_data_save(cls, data):
        """Save the global summary cache."""
        global g_summary_data_id
        data_id = g_summary_data_id
        save_summary_data(data_id, data)
        g_summary_data_id += 1
        return data_id

    @classmethod
    def summary_file_set(cls, event_writer):
        """Support the many event_writer."""
        global g_summary_file, g_summary_writer_id
        g_summary_writer_id += 1
        g_summary_file[g_summary_writer_id] = event_writer
        return g_summary_writer_id

    @classmethod
    def summary_file_get(cls, writer_id=1):
        ret = None
        global g_summary_file
        if writer_id in g_summary_file:
            ret = g_summary_file.get(writer_id)
        return ret


class WorkerScheduler:
    """
    Create worker and schedule data to worker.

    Args:
        writer_id (int): The index of writer.
    """
    def __init__(self, writer_id):
        # Create the process of write event file
        self.write_lock = mp.Lock()
        # Schedule info for all worker
        # Format: {worker: (step, WorkerStatus)}
        self.schedule_table = {}
        # write id
        self.writer_id = writer_id
        self.has_graph = False

    def dispatch(self, step, data):
        """
        Select schedule strategy and dispatch data.

        Args:
            step (Number): The number of step index.
            data (Object): The data of recode for summary.

        Retruns:
            bool, run successfully or not.
        """
        # save the data to global cache , convert the tensor to numpy
        result, size, data = self._data_convert(data)
        if result is False:
            logger.error("The step(%r) summary data(%r) is invalid.", step, size)
            return False

        data_id = SummaryDataManager.summary_data_save(data)
        self._start_worker(step, data_id)
        return True

    def _start_worker(self, step, data_id):
        """
        Start worker.

        Args:
            step (Number): The index of recode.
            data_id (str): The id of work.

        Return:
            bool, run successfully or not.
        """
        # assign the worker
        policy = self._make_policy()
        if policy == ScheduleMethod.TEMP_WORKER:
            worker = SummaryDataProcess(step, data_id, self.write_lock, self.writer_id)
            # update the schedule table
            self.schedule_table[worker] = (step, data_id, WorkerStatus.WORKER_INIT)
            # start the worker
            worker.start()
        else:
            logger.error("Do not support the other scheduler policy now.")

        # update the scheduler infor
        self._update_scheduler()
        return True

    def _data_convert(self, data_list):
        """Convert the data."""
        if data_list is None:
            logger.warning("The step does not have record data.")
            return False, 0, None

        # convert the summary to numpy
        size = 0
        for v_dict in data_list:
            tag = v_dict["name"]
            data = v_dict["data"]
            # confirm the data is valid
            summary_type, summary_tag = _parse_tag_format(tag)
            if summary_type == SummaryType.INVALID:
                logger.error("The data type is invalid, tag = %r, tensor = %r", tag, data)
                return False, 0, None
            if isinstance(data, Tensor):
                # get the summary type and parse the tag
                v_dict["name"] = summary_tag
                v_dict["type"] = summary_type
                v_dict["data"] = data.asnumpy()
                size += v_dict["data"].size
            else:
                logger.error("The data type is invalid, tag = %r, tensor = %r", tag, data)
                return False, 0, None

        return True, size, data_list

    def _update_scheduler(self):
        """Check the worker status and update schedule table."""
        workers = list(self.schedule_table.keys())
        for worker in workers:
            if not worker.is_alive():
                # update the table
                worker.join()
                del self.schedule_table[worker]

    def close(self):
        """Confirm all worker is end."""
        workers = self.schedule_table.keys()
        for worker in workers:
            if worker.is_alive():
                worker.join()

    def _make_policy(self):
        """Select the schedule strategy by data."""
        # now only support the temp worker
        return ScheduleMethod.TEMP_WORKER


class SummaryDataProcess(mp.Process):
    """
    Process that consume the summarydata.

    Args:
        step (int): The index of step.
        data_id (int): The index of summary data.
        write_lock (Lock): The process lock for writer same file.
        writer_id (int): The index of writer.
    """
    def __init__(self, step, data_id, write_lock, writer_id):
        super(SummaryDataProcess, self).__init__()
        self.daemon = True
        self.writer_id = writer_id
        self.writer = SummaryDataManager.summary_file_get(self.writer_id)
        if self.writer is None:
            logger.error("The writer_id(%r) does not have writer", writer_id)
        self.step = step
        self.data_id = data_id
        self.write_lock = write_lock
        self.name = "SummaryDataConsumer_" + str(self.step)

    def run(self):
        """The consumer is process the step data and exit."""
        # convert the data to event
        # All exceptions need to be caught and end the queue
        try:
            logger.debug("process(%r) process a data(%r)", self.name, self.step)
            # package the summary event
            summary_event = package_summary_event(self.data_id, self.step)
            # send the event to file
            self._write_summary(summary_event)
        except Exception as e:
            logger.error("Summary data mq consumer exception occurred, value = %r", e)

    def _write_summary(self, summary_event):
        """
        Write the summary to event file.

        Note:
            The write record format:
            1 uint64 : data length.
            2 uint32 : mask crc value of data length.
            3 bytes  : data.
            4 uint32 : mask crc value of data.

        Args:
            summary_event (Event): The summary event of proto.

        """
        event_str = summary_event.SerializeToString()
        self.write_lock.acquire()
        self.writer.write_event_to_file(event_str)
        self.writer.flush()
        self.write_lock.release()
