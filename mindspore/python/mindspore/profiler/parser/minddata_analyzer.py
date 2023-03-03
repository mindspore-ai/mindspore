# Copyright 2021 Huawei Technologies Co., Ltd
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
"""The analyzer for MindData profiling files."""
import copy
import csv
import json
import os
import stat

from mindspore.profiler.common.exceptions.exceptions import \
    ProfilerPathErrorException, ProfilerFileNotFoundException, \
    ProfilerDirNotFoundException, ProfilerRawFileException
from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


class MinddataProfilingAnalyzer:
    """
    The analyzer for MindData profiling files.

    Args:
        source_dir (str): The source directory for MindData profiling input files.
        device_id (str): The device ID.
        output_path (str): The target directory for the analyzed summary. Default: `./`.

    Raises:
        ProfilerPathErrorException: If the source directory or the output path is invalid.
        ProfilerDirNotFoundException: If the source directory or the output path does not exist.
        ProfilerFileNotFoundException: If any of the MindData profiling input files do not exist.
    """

    def __init__(self, source_dir, device_id, output_path='./'):
        # Validate and save input parameters
        self._device_id = device_id
        self._source_dir = self._validate_directory(source_dir, 'Source directory')
        self._output_path = self._validate_directory(output_path, 'Output path')

        # Get MindData profiling input filenames
        self._pipeline_path_filename = self._get_pipeline_path_filename(source_dir)
        self._cpu_utilization_path_filename = self._get_cpu_utilization_path_filename(source_dir)
        self._device_trace_path_filename, self._device_queue_file_found = \
            self._get_device_trace_path_filename(source_dir)

        # Save output filename
        self._save_path = self._get_save_path(output_path)

    @property
    def save_path(self):
        """
        The property of save path.

        Returns:
            str, the save path.
        """
        return self._save_path

    @staticmethod
    def _validate_directory(dir_name, dir_type):
        """
        Validate the input directory.

        Args:
             dir_name (str): The directory name.
             dir_type (str): The type of directory.  (Should begin with capital since is used for output messages.)
        """
        try:
            validated_dir = validate_and_normalize_path(dir_name)
        except RuntimeError as path_error:
            logger.warning('<%s> is invalid.', dir_type)
            raise ProfilerPathErrorException(dir_type + ' is invalid.') from path_error

        if not os.path.isdir(validated_dir):
            logger.warning('<%s> <%s> not found.', dir_type, validated_dir)
            raise ProfilerDirNotFoundException(validated_dir)
        return validated_dir

    @staticmethod
    def _parse_pipeline_metrics_info(metrics):
        """
        Parse and process the pipeline profiling metrics information for a given op.

        Args:
            metrics (dict): The pipeline profiling metrics information for a given op.

        Returns:
            List with the following analyzed metrics information:
                output queue size
                output queue length
                output queue average size,
                output queue utilization percentage
                output queue empty frequency percentage
        """
        # Note: Some ops like DeviceQueue and inline ops do not have metrics information
        queue_size = -1
        queue_length = -1
        queue_average_size = -1
        queue_utilization_pct = -1
        queue_empty_freq_pct = -1
        if metrics and metrics['output_queue']:
            queue_size = metrics['output_queue']['size']
            queue_length = metrics['output_queue']['length']
            queue_average_size = round(sum(queue_size) / len(queue_size), 2) if queue_size else -1
            queue_utilization_pct = round(100 * queue_average_size / queue_length, 2) if queue_length else -1
            # Compute percentage of time queue is empty
            empty_count = 0
            for q_size in queue_size:
                if q_size == 0:
                    empty_count += 1
            queue_empty_freq_pct = round(100 * empty_count / len(queue_size), 2) if queue_size else -1
        return [queue_size, queue_length, queue_average_size, queue_utilization_pct, queue_empty_freq_pct]

    @staticmethod
    def _parse_cpu_util_info(cpu_util_info):
        """
        Parse and process the CPU profiling information.

        Args:
            cpu_util_info (dict): The CPU utilization profiling information.

        Returns:
            Dictionary with analyzed summary output information
            Dictionary consists of:
                avg_cpu_pct: Average CPU utilization percentage for each op, a list ordered by increasing op id

        Raises:
            ProfilerRawFileException: If the format of the input is wrong.
        """
        # Perform sanity checks for CPU utilization information
        cpu_processor_num = cpu_util_info.get('cpu_processor_num')
        cpu_op_info = cpu_util_info.get('op_info')
        if cpu_processor_num is None or not cpu_op_info:
            raise ProfilerRawFileException('The format of MindData CPU utilization JSON file is wrong.')

        for item in cpu_op_info:
            if not item:
                raise ProfilerRawFileException('The contents of MindData CPU utilization JSON file is wrong.')

        # Parse and process the following CPU utilization information:
        # - overage cpu utilization for each op
        dict_opid_cpuutil = {}
        for op in cpu_util_info["op_info"]:
            # Note: The CPU utilization data may have an extra entry with op_id=-1
            # Omit info for op_id=1
            if op["op_id"] != -1:
                op_sys, op_usr = op["metrics"]["sys_utilization"], op["metrics"]["user_utilization"]
                dict_opid_cpuutil[op["op_id"]] = [op_sys[i] + op_usr[i] for i in range(len(op_sys))]

        # Initialize oplist_avg_cpu_pct with -1 for each pipeline op, since
        # CPU utilization data may not have information for each pipeline op
        oplist_avg_cpu_pct = [-1] * len(dict_opid_cpuutil)
        total_cpu = 0
        for op_id, cpu in dict_opid_cpuutil.items():
            op_avg_cpu_pct = sum(cpu) / len(cpu) if cpu else 0
            oplist_avg_cpu_pct[op_id] = round(op_avg_cpu_pct, 2)
            total_cpu += op_avg_cpu_pct

        return_dict = {}
        return_dict['avg_cpu_pct'] = oplist_avg_cpu_pct
        return return_dict

    @staticmethod
    def _compute_composite_info(summary_dict):
        """
        Compute composite analysis information from the current summary pipeline data.

        Args:
            summary_dict (dict): Input summary pipeline information.

        Returns:
            Dictionary with composite analysis output information
            Dictionary consists of:
                avg_cpu_pct_per_worker: Average CPU utilization percentage per worker
        """
        return_dict = {}

        # Build list: average CPU utilization percentage per worker - for each op
        avg_cpu_pct_per_worker = []
        for c, n in zip(summary_dict.get('avg_cpu_pct'), summary_dict.get('num_workers')):
            avg_cpu_pct_per_worker.append(round(c / n if (n != 0 and c >= 0) else -1, 2))
        return_dict['avg_cpu_pct_per_worker'] = avg_cpu_pct_per_worker

        return return_dict

    @staticmethod
    def _analyze_for_bottleneck_op(summary_dict):
        """
        Analyze the MindData summary information and identify any potential bottleneck operator
        in the MindData pipeline.

        Args:
            summary_dict (dict): Input summary pipeline information.

        Returns:
            Dictionary with the following information, if applicable:
            - CPU utilization analysis
            - queue utilization analysis
            - bottleneck warning: Information on the bottleneck op
                (This is returned only if a potential bottleneck is identified.)
            - bottleneck suggestion: Reason why the subject op is it is identified as
                a potential bottleneck, plus suggestion on how to resolve the bottleneck.
                (This is returned only if a potential bottleneck is identified.)
        """
        try:
            bottleneck_analyzer = BottleneckAnalyzer(summary_dict)
            return_dict = bottleneck_analyzer.analyze()
        except IndexError:
            return_dict = {}

        return return_dict

    def analyze(self):
        """
        Analyze the MindData profiling files, produce summary pipeline information, including potential
        bottleneck operator in the MindData pipeline, and save the result to disk.

        Returns:
            dict, Analyzed MindData pipeline summary information, which is also written to disk in
               JSON file 'minddata_pipeline_summary_<device_id>.json' and
               CSV file 'minddata_pipeline_summary_<device_id>.csv'.

        Raises:
            ProfilerRawFileException: If fails to find a MindData profiling file or a file is empty.
        """

        # Open the MindData pipeline file
        with open(self._pipeline_path_filename, 'r') as pipeline_file:
            try:
                pipeline_info = json.load(pipeline_file)
            except (json.JSONDecodeError, TypeError) as path_filename_error:
                logger.warning(path_filename_error)
                raise ProfilerRawFileException(
                    'Failed to find the MindData pipeline profiling file.') from path_filename_error
        if not pipeline_info:
            logger.warning('The MindData pipeline file <%s> is empty.', self._pipeline_path_filename)
            raise ProfilerRawFileException('The MindData pipeline file is empty.')

        # Open the CPU utilization file
        with open(self._cpu_utilization_path_filename, 'r') as cpu_util_file:
            try:
                cpu_util_info = json.load(cpu_util_file)
            except (json.JSONDecodeError, TypeError) as path_filename_error:
                logger.warning(path_filename_error)
                raise ProfilerRawFileException(
                    'Failed to find the MindData CPU utilization file.') from path_filename_error
        if not cpu_util_info:
            logger.warning('The MindData CPU utilization file <%s> is empty.', self._cpu_utilization_path_filename)
            raise ProfilerRawFileException('The MindData CPU utilization file is empty.')

        # Open the device queue or dataset iterator trace profiling file
        with open(self._device_trace_path_filename, 'r') as device_trace_file:
            try:
                device_trace_info = device_trace_file.readlines()
            except (TypeError) as path_filename_error:
                logger.warning(path_filename_error)
                raise ProfilerRawFileException(
                    'Failed to find the MindData trace profiling file.') from path_filename_error
        if not device_trace_info:
            logger.warning('The MindData trace profiling file <%s> is empty.', self._device_trace_path_filename)
            raise ProfilerRawFileException('The MindData trace profiling file is empty.')

        # Analyze the MindData profiling file information and save the result
        summary_dict = self._analyze_and_save(pipeline_info, cpu_util_info, device_trace_info)
        return summary_dict

    def _get_pipeline_path_filename(self, source_dir):
        """
        Get the MindData pipeline full path filename.
        The filename is 'pipeline_profiling_<device_id>.json'.

        Args:
            source_dir (str): The source directory for MindData profiling files.

        Returns:
            str, the MindData pipeline full path filename.
        """

        pipeline_profiling_templatename = 'pipeline_profiling_{}.json'
        pipeline_path_filename = os.path.join(
            source_dir,
            pipeline_profiling_templatename.format(self._device_id))

        try:
            pipeline_path_filename = validate_and_normalize_path(pipeline_path_filename)
        except RuntimeError as path_filename_error:
            logger.warning('The MindData pipeline path %s is invalid.', pipeline_path_filename)
            raise ProfilerPathErrorException('The MindData pipeline path is invalid.') from path_filename_error

        if not os.path.isfile(pipeline_path_filename):
            logger.warning('The MindData pipeline file <%s> is not found.', pipeline_path_filename)
            raise ProfilerFileNotFoundException(pipeline_path_filename)

        return pipeline_path_filename

    def _get_cpu_utilization_path_filename(self, source_dir):
        """
        Get the MindData CPU utilization full path filename.
        The filename is 'minddata_cpu_utilization_<device_id>.json'.

        Args:
            source_dir (str): The source directory for MindData profiling files.

        Returns:
            str, the MindData CPU utilization full path filename.
        """
        cpu_utilization_templatename = 'minddata_cpu_utilization_{}.json'
        cpu_utilization_path_filename = os.path.join(
            source_dir,
            cpu_utilization_templatename.format(self._device_id))

        try:
            cpu_utilization_path_filename = validate_and_normalize_path(cpu_utilization_path_filename)
        except RuntimeError as path_filename_error:
            logger.warning('The MindData CPU utilization path <%s> is invalid.', cpu_utilization_path_filename)
            raise ProfilerPathErrorException('The MindData CPU utilization path is invalid.') from path_filename_error

        if not os.path.isfile(cpu_utilization_path_filename):
            logger.warning('The MindData CPU utilization file <%s> is not found.', cpu_utilization_path_filename)
            raise ProfilerFileNotFoundException(cpu_utilization_path_filename)

        return cpu_utilization_path_filename

    def _get_device_trace_path_filename(self, source_dir):
        """
        Get the MindData device trace profiling full path filename.
        File search order:
        1) 'device_queue_profiling_<device_id>.txt' and then
        2) 'dataset_iterator_profiling_<device_id>.txt'.

        Args:
            source_dir (str): The source directory for MindData profiling files.

        Returns:
            str, the MindData device trace profiling full path filename.
            bool, flag which indicates if 'device_queue_profiling_<device_id>.txt' has been found or not
        """
        # Initialize variable for MindData device trace profiling filename
        device_trace_path_filename = ''
        # Initialize flag that 'device_queue_profiling_<device_id>.txt' has not yet been found
        device_queue_file_found = False

        txt_names = [os.path.join(source_dir, txt_name.format(self._device_id))
                     for txt_name in ('device_queue_profiling_{}.txt', 'dataset_iterator_profiling_{}.txt')]

        # Search for a device trace profiling file
        if os.path.exists(txt_names[0]):
            device_trace_path_filename = txt_names[0]
            device_queue_file_found = True
        elif os.path.exists(txt_names[1]):
            device_trace_path_filename = txt_names[1]
        else:
            logger.warning('A MindData device trace profiling file <%s> nor <%s> cannot be found.',
                           txt_names[0], txt_names[1])
            raise ProfilerPathErrorException('A MindData device trace profiling file cannot be found.')

        if not os.path.isfile(device_trace_path_filename):
            logger.warning('The MindData device trace profiling file <%s> is not found.', device_trace_path_filename)
            raise ProfilerFileNotFoundException(device_trace_path_filename)

        return device_trace_path_filename, device_queue_file_found

    def _get_save_path(self, output_path):
        """
        Get the full pathname for the output file to save MindData pipeline summary analyzed information.
        The output filename is 'minddata_pipeline_summary_<device_id>.json'.

        Args:
            output_path (str): The output directory.

        Returns:
            str, the save path.
        """
        try:
            output_dir = validate_and_normalize_path(output_path)
        except RuntimeError as path_error:
            logger.warning('Output path <%s> is invalid.', output_path)
            raise ProfilerPathErrorException('Output path is invalid.') from path_error

        if not os.path.isdir(output_dir):
            logger.warning('The output directory <%s> not found.', output_dir)
            raise ProfilerDirNotFoundException(output_dir)

        summary_templatename = 'minddata_pipeline_summary_{}.json'
        return os.path.join(output_dir, summary_templatename.format(self._device_id))

    def _parse_pipeline_info(self, pipeline_info):
        """
        Parse and process the pipeline profiling information.

        Args:
            pipeline_info (dict): The pipeline profiling information.

        Returns:
            Dictionary with analyzed summary output information
            For the following key-value pairs, each value is a list ordered by increasing op id
                pipeline_ops: operator name and operator id, a string, with example format Batch(id=0)
                op_names: operator name, a string
                op_ids: operator id, an integer
                num_workers: number of workers for the op, an integer
                queue_average_size: average queue size for the op, a float
                queue_utilization_pct: average percentage of time queue is used for op, a float from 0.00 to 1.00
                queue_empty_freq_pct: percentage of time queue is empty for op, a float from 0.00 to 1.00
                children_ids: children op ids of op; list if empty [] if op has no children
                parent_id: parent id of op

        Raises:
            ProfilerRawFileException: If the format of the input is wrong.
        """
        # Perform sanity checks for pipeline information
        pipeline_op_info = pipeline_info.get('op_info')
        for item in pipeline_op_info:
            if not item:
                raise ProfilerRawFileException('The contents of MindData pipeline JSON file is wrong.')

        # Parse and process pipeline information
        # Obtain the following for each op (and build a list), ordered by increasing op id
        # - op id (handy for user output)
        # - op name (needed for basic processing)
        # - op name with op id (handy for user output)
        # - num_workers
        # - various queue information
        # - children op ids
        # - parent op id
        dict_opid_pipeline_ops = {}
        dict_opid_opname = {}
        dict_opid_numworkers = {}
        dict_opid_queue_info = {}
        dict_opid_children_ids = {}
        dict_opid_parent_id = {}
        # Note: Will process the input pipeline ops in "reversed" order since typically they are ordered
        #       from largest op id (usually leaf/source op) to smallest op id (usually root).
        #       However, since there may be non-linear pipelines, the processed op info needs to be sorted
        #       before final output is produced and saved.
        for op_info in reversed(pipeline_info['op_info']):
            op_id = op_info.get('op_id')
            op_name = op_info.get('op_type')[0:-2]
            dict_opid_pipeline_ops[op_id] = '{}(id={})'.format(op_name, str(op_id))
            dict_opid_opname[op_id] = op_name
            dict_opid_numworkers[op_id] = op_info.get('num_workers')

            # Obtain the output queue metrics information for the current op
            dict_opid_queue_info[op_id] = self._parse_pipeline_metrics_info(op_info.get('metrics'))

            # For current op, initialize parent_id=-1, in case after processing all children in pipeline,
            # it is determined that current op has no parent
            if dict_opid_parent_id.get(op_id) is None:
                dict_opid_parent_id[op_id] = -1

            children_ids = op_info.get('children')
            if children_ids:
                # Set children op ids for current op
                dict_opid_children_ids[op_id] = children_ids
                # For each child op, set parent op to be current op
                for child_op_id in children_ids:
                    dict_opid_parent_id[child_op_id] = op_id
            else:
                dict_opid_children_ids[op_id] = []

        # Build resultant dictionary
        return_dict = {}

        return_dict['pipeline_ops'] = [x[1] for x in sorted(dict_opid_pipeline_ops.items())]
        return_dict['op_names'] = [x[1] for x in sorted(dict_opid_opname.items())]
        return_dict['op_ids'] = sorted(dict_opid_opname.keys())
        return_dict['num_workers'] = [x[1] for x in sorted(dict_opid_numworkers.items())]

        queue_info_items = [x[1] for x in sorted(dict_opid_queue_info.items())]
        return_dict['queue_average_size'] = [y[2] for y in queue_info_items]
        return_dict['queue_utilization_pct'] = [y[3] for y in queue_info_items]
        return_dict['queue_empty_freq_pct'] = [y[4] for y in queue_info_items]

        return_dict['children_ids'] = [x[1] for x in sorted(dict_opid_children_ids.items())]
        return_dict['parent_id'] = [x[1] for x in sorted(dict_opid_parent_id.items())]

        return return_dict

    def _parse_device_trace_info(self, device_trace_info):
        """
        Parse and process the device trace profiling information.

        Args:
            device_trace_info: The device trace profiling information in text format, one line per record.

        Returns:
            Dictionary with analyzed summary output information
            Dictionary consists of:
                per_batch_time: Average per batch time for pipeline in milliseconds
                per_pipeline_time: Average per pipeline time in milliseconds
                per_push_queue_time: Average per queue push time in milliseconds
        """
        # Information on the format of the device tracing profiling information.
        # Format is: type extra-info batch-num value timestamp
        # 0) type: 0: time,  1: connector size
        # 1) extra-info: if type is 0 - 0: pipeline time, 1: push tdt time, 2: batch time
        #                if type is 1 - connector capacity
        # 2) batch-num: batch number
        # 3) value: if type is 0 - value is time(ms)
        #           if type is 1 - value is connector size
        # 4) timestamp
        # Examples:
        # 0 0 20 10 xxx - The 20th batch took 10ms to get data from pipeline.
        # 1 64 20 5 yyy - Connector size is 5 when get the 20th batch.Connector capacity is 64.

        prev_time = 0
        q_time = [[], [], []]  # pipeline time, push TDT time, batch time

        # Parse each record
        for line_data in device_trace_info:
            record = [int(d) for d in line_data.split(" ")][0:5]
            if record[2] < 2:  # skip 1st batch
                prev_time = record[4]
                continue

            if record[0] == 0:  # type 0: time record
                q_time[record[1]].append(record[3])
            elif record[0] == 1:  # type 1: connector size record
                # Check if dataset_iterator trace profiling file was found
                if not self._device_queue_file_found:
                    q_time[2].append(record[4] - prev_time)
                    prev_time = record[4]

        # Compute average queue times
        avg_pipeline_time = sum(q_time[0]) / len(q_time[0]) if q_time[0] else -1
        avg_push_queue_time = sum(q_time[1]) / len(q_time[1]) if q_time[1] else -1
        avg_batch_time = sum(q_time[2]) / len(q_time[2]) if q_time[2] else -1

        return_dict = {}
        return_dict['per_batch_time'] = [round(avg_batch_time, 3)]
        return_dict['per_pipeline_time'] = [round(avg_pipeline_time, 3)]
        return_dict['per_push_queue_time'] = [round(avg_push_queue_time, 3)]

        return return_dict

    def _save_as_csv_file(self, data_dict):
        """
        Save data dictionary information to CSV file.

        Args:
            data_dict (dict): Input data dictionary information.

        Returns:
            Data dictionary information is saved to CSV file named 'minddata_pipeline_summary_<device_id>.csv'.
        """

        summary_templatename = 'minddata_pipeline_summary_{}.csv'
        output_csv_path_filename = os.path.join(self._output_path, summary_templatename.format(self._device_id))

        # Open file for writing
        data_file = open(output_csv_path_filename, 'w')

        # Create CSV writer object
        csv_writer = csv.writer(data_file)

        # Write the dictionary information to CSV file
        # Create deepcopy of input data_dict so zip processing in this function does NOT change the data_dict
        temp_dict = copy.deepcopy(data_dict)
        for data_key, data_value in zip(temp_dict.keys(), temp_dict.values()):
            # Begin/prefix the data value with the data key
            data_value.insert(0, data_key)
            csv_writer.writerow(data_value)

        # Close file for writing
        data_file.close()

        # Update file permissions
        os.chmod(output_csv_path_filename, stat.S_IREAD | stat.S_IWRITE)

    def _analyze_and_save(self, pipeline_info, cpu_util_info, device_trace_info):
        """
        Analyze and save the MindData summary information to file.

        Args:
            pipeline_info (dict): The pipeline information read from the input JSON file.
            cpu_util_info (dict): The CPU utilization information read from the input JSON file.
            device_trace_info (text): The dataset iterator (CPU) or device queue (GPU, Ascend) trace profiling
                                     text file. Value is None if such file could not be identified.

        Returns:
            summary_dict (dict): Analyzed summary information.
            The summary dictionary information is doubly saved to a JSON file and a CSV file
            (so that these different formats are available to the users).
        """

        # Initialize summary output dictionary
        summary_dict = {}

        # Parse and process pipeline information
        summary_dict.update(self._parse_pipeline_info(pipeline_info))

        # Parse and process CPU utilization information
        summary_dict.update(self._parse_cpu_util_info(cpu_util_info))

        if device_trace_info is not None:
            # Parse and process device queue or dataset iterator trace profiling information
            summary_dict.update(self._parse_device_trace_info(device_trace_info))

        # Check if both pipeline data and CPU utilization data have the same number of ops
        num_pipeline_ops = len(summary_dict.get('pipeline_ops'))
        num_cpu_util_ops = len(summary_dict.get('avg_cpu_pct'))
        if num_pipeline_ops == num_cpu_util_ops:
            # Compute composite analysis information
            summary_dict.update(self._compute_composite_info(summary_dict))

            # Analyze pipeline info for potential bottleneck op
            bottleneck_dict = self._analyze_for_bottleneck_op(summary_dict)
            if bottleneck_dict:
                summary_dict.update(bottleneck_dict)

        else:
            # Produce a warning since the pipeline data and the CPU utilization data do not include information
            # for the same number of ops
            warning_msg = 'Number of ops for pipeline data: ' + str(num_pipeline_ops) + \
                          ' does not match number of ops for CPU utilization data: ' + str(num_cpu_util_ops)
            logger.warning(warning_msg)

        # Save summary output dictionary to JSON output file (format#1)
        with open(self._save_path, 'w') as save_file:
            json.dump(summary_dict, save_file)

        os.chmod(self._save_path, stat.S_IREAD | stat.S_IWRITE)

        # Save summary output to CSV file (format#2)
        self._save_as_csv_file(summary_dict)
        # Return summary output dictionary (format#3)
        return summary_dict


class BottleneckAnalyzer:
    """ analyzer for bottleneck """

    # These are the threshold values used in the pipeline bottleneck analyzer algorithm
    _AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM = 75.0
    _AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM = 20.0
    _LEAF_OUTPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM = 50
    _DEVICEQUEUE_INPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM = 60
    _IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM = 50
    _IN_QUEUE_UTIL_PCT_MAXIMUM = 10

    def __init__(self, summary_dict):
        """ constructor for BottleneckAnalyzer """
        self.pipeline_ops = summary_dict["pipeline_ops"]
        self.op_names = summary_dict["op_names"]
        self.op_ids = summary_dict["op_ids"]
        self.num_workers = summary_dict["num_workers"]
        self.queue_average_size = summary_dict["queue_average_size"]
        self.queue_utilization_pct = summary_dict["queue_utilization_pct"]
        self.queue_empty_freq_pct = summary_dict["queue_empty_freq_pct"]
        self.children_ids = summary_dict["children_ids"]
        self.parent_id = summary_dict["parent_id"]
        self.avg_cpu_pct = summary_dict["avg_cpu_pct"]
        self.avg_cpu_pct_per_worker = summary_dict["avg_cpu_pct_per_worker"]

        self.op_id_not_exist = -1
        self.queue_usage_not_exist = -1
        self.non_multithreaded_ops = set(["Barrier",
                                          "Concat",
                                          "EpochCtrl",
                                          "Rename",
                                          "Repeat",
                                          "Shuffle",
                                          "Skip",
                                          "Take",
                                          "Zip"])

    def analyze(self):
        """ analyze all op's usage """
        detailed_analysis = {}

        cpu_analysis = self.analyze_cpu_usage()
        queue_analysis = self.analyze_queue_usage()

        if cpu_analysis:
            detailed_analysis["cpu_analysis_details"] = cpu_analysis

        if queue_analysis:
            detailed_analysis["queue_analysis_details"] = queue_analysis

        bottleneck, suggestion = self.analyze_bottleneck()

        if bottleneck[0]:
            detailed_analysis["bottleneck_warning"] = bottleneck
            detailed_analysis["bottleneck_suggestion"] = suggestion

        return detailed_analysis

    def analyze_cpu_usage(self):
        """ analyze cpu usage of each op """
        cpu_usage_analysis = []
        for op_id in self.op_ids:
            if op_id == self.op_id_not_exist or self.op_names[op_id] in self.non_multithreaded_ops:
                continue

            if self.avg_cpu_pct_per_worker[op_id] > self._AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM and \
                    self.op_names[op_id]:
                cpu_usage_analysis.append(
                    ("{} is using {}% CPU per worker."
                     " Setting num_parallel_workers"
                     ">{} might bring extra performance.").format(self.pipeline_ops[op_id],
                                                                  self.avg_cpu_pct_per_worker[op_id],
                                                                  self.num_workers[op_id]))
            elif self.avg_cpu_pct_per_worker[op_id] < self._AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM and \
                    self.num_workers[op_id] > 1:
                cpu_usage_analysis.append(
                    ("{} is using {}% CPU per worker. Using num_parallel_workers={} might not bring as much benefit"
                     " due to low CPU usage per worker.").format(self.pipeline_ops[op_id],
                                                                 self.avg_cpu_pct_per_worker[op_id],
                                                                 self.num_workers[op_id]))
        return cpu_usage_analysis

    def analyze_queue_usage(self):
        """ analyze queue usage of each op """
        queue_usage_analysis = []
        for op_id in self.op_ids:
            if op_id == self.op_id_not_exist or self.op_names[op_id] in self.non_multithreaded_ops:
                continue

            if self.op_names[op_id] == "Batch":
                continue
            in_op_id, out_q = self.__get_non_inline_child_recur(
                op_id), self.queue_utilization_pct[op_id]
            if in_op_id == self.op_id_not_exist and out_q != self.queue_usage_not_exist:
                # This is a leaf node since input queue does not exist and output queue exists
                if out_q < self._LEAF_OUTPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM:
                    queue_usage_analysis.append(("Leaf op {} is using {}% of its output queue."
                                                 "Setting num_parallel_workers"
                                                 ">{} might speed up I/O.").format(self.pipeline_ops[op_id],
                                                                                   out_q,
                                                                                   self.num_workers[op_id]))
            elif self.op_names[op_id] == "DeviceQueue" and in_op_id != self.op_id_not_exist:
                # if this is device_queue op,
                if self.queue_empty_freq_pct[in_op_id] > self._DEVICEQUEUE_INPUT_QUEUE_EMPTY_FREQ_PCT_MAXIMUM:
                    queue_usage_analysis.append(
                        f"{self.pipeline_ops[op_id]}'s input queue is empty {self.queue_empty_freq_pct[in_op_id]}% "
                        f"of the time. This might indicate dataset bottlenecks. Hence host cannot keep up with "
                        f"the device {self.queue_empty_freq_pct[in_op_id]}% of the time. "
                        f"Device waits whenever input queue is empty.")
            elif in_op_id != self.op_id_not_exist and out_q != self.queue_usage_not_exist:
                in_q = self.queue_utilization_pct[in_op_id]
                if in_q != self.queue_usage_not_exist and in_q - out_q > self._IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM:
                    queue_usage_analysis.append(
                        f"{self.pipeline_ops[op_id]}'s input queue usage={in_q}% is greater output queue "
                        f"usage={out_q}%. This indicates child op {self.pipeline_ops[in_op_id]} "
                        f"might be producing faster than its parent {self.pipeline_ops[op_id]} can consume. "
                        f"If this op has low CPU utilization, try increasing "
                        f"prefetch_size or increasing num_workers.")
        return queue_usage_analysis

    def analyze_bottleneck(self):
        """ analyze bottleneck by using both cpu and queue usage """
        bottleneck, suggestion = "", ""
        for op_id in reversed(self.op_ids):
            in_op_id, out_q = self.__get_non_inline_child_recur(
                op_id), self.queue_utilization_pct[op_id]
            wkr_cpu = self.avg_cpu_pct_per_worker[op_id]
            if op_id == self.op_id_not_exist or \
                    self.op_names[op_id] in self.non_multithreaded_ops \
                    or self.op_names[op_id] == "DeviceQueue":
                continue

            if wkr_cpu > self._AVG_CPU_UTIL_PCT_PER_WORKER_MAXIMUM:
                bottleneck = self.pipeline_ops[op_id]
                suggestion = "{} has high CPU utilization per worker of {}%".format(
                    self.pipeline_ops[op_id], wkr_cpu)
                suggestion += " Try increasing num_parallel_workers above {}.".format(self.num_workers[op_id])
            elif wkr_cpu < self._AVG_CPU_UTIL_PCT_PER_WORKER_MINIMUM:
                in_op_id = self.__get_non_inline_child_recur(op_id)
                in_q_usage = self.queue_utilization_pct[in_op_id]
                if in_op_id != self.op_id_not_exist and (
                        in_q_usage < self._IN_QUEUE_UTIL_PCT_MAXIMUM or out_q -
                        in_q_usage > self._IN_OUT_QUEUE_UTIL_PCT_DIFF_MAXIMUM):
                    bottleneck = self.pipeline_ops[op_id]
                    suggestion = "{} has low CPU utilization per worker of {}%".format(
                        self.pipeline_ops[op_id], wkr_cpu)
                    suggestion += " and abnormal queue usage. Try increasing prefetch_size."

        return [bottleneck], [suggestion]

    def __get_non_inline_child_recur(self, cur_op_id):
        """get the child id of cur op which isn't an inline op"""
        if cur_op_id == self.op_id_not_exist or not self.children_ids[cur_op_id]:
            return self.op_id_not_exist
        cur_child_id = self.children_ids[cur_op_id][0]
        if self.queue_average_size[cur_child_id] != -1:
            return cur_child_id
        return self.__get_non_inline_child_recur(cur_child_id)
