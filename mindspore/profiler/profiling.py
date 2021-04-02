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
"""Profiling api file."""
import os
import re
import stat
import time
import json
from enum import Enum

from mindspore import log as logger, context
from mindspore.communication.management import GlobalComm, release, get_rank
import mindspore._c_expression as c_expression
from mindspore.dataset.core.config import _stop_dataset_profiler
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException, \
    ProfilerIOException, ProfilerException, ProfilerRawFileException
from mindspore.profiler.common.util import get_file_names, fwrite_format
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path
from mindspore.profiler.parser.aicpu_data_parser import DataPreProcessParser
from mindspore.profiler.parser.framework_parser import FrameworkParser
from mindspore.profiler.parser.hwts_log_parser import HWTSLogParser
from mindspore.profiler.parser.integrator import Integrator
from mindspore.profiler.parser.integrator import GpuTimelineGenerator, AscendTimelineGenerator
from mindspore.profiler.parser.memory_usage_parser import MemoryUsageParser
from mindspore.profiler.parser.minddata_parser import MinddataParser
from mindspore.profiler.parser.minddata_pipeline_parser import \
    MinddataPipelineParser
from mindspore.profiler.parser.optime_parser import OPComputeTimeParser
from mindspore.profiler.parser.step_trace_parser import GpuStepTraceParser, AscendStepTraceParser
from mindspore.nn.cell import Cell

INIT_OP_NAME = 'Default/InitDataSetQueue'

class ProfileOption(Enum):
    """
    Profile Option Enum which be used in Profiler.profile.
    """
    trainable_parameters = 0

class Profiler:
    """
    Performance profiling API.

    This API enables MindSpore users to profile the performance of neural network.
    Profiler supports Ascend and GPU, both of them are used in the same way,
    but only output_path in args works on GPU.

    Args:
        output_path (str): Output data path.
        optypes_not_deal (str): (Ascend only) Op type names, determine the data of which optype should be collected
            and analysed,will deal with all op if null; Different op types should be separated by comma.
        ascend_job_id (str): (Ascend only) The directory where the profiling files to be parsed are located;
            This parameter is used to support offline parsing.

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, context
        >>> from mindspore.train import Model
        >>> import mindspore.dataset as ds
        >>> from mindspore.profiler import Profiler
        >>>
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2,2)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator():
        ...     for i in range(2):
        ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
        >>>
        >>> def train(net):
        ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator, ["data", "label"])
        ...     model = Model(net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is GPU, set the device_target to "GPU"
        ...     context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        ...
        ...     # Init Profiler
        ...     # Note that the Profiler should be initialized after context.set_context and before model.train
        ...     profiler = Profiler()
        ...
        ...     # Train Model
        ...     net = Net()
        ...     train(net)
        ...
        ...     # Profiler end
        ...     profiler.analyse()
    """

    _hwts_output_filename_target = "output_format_data_hwts_"
    _opcompute_output_filename_target = "output_op_compute_time_"
    _aicpu_op_output_filename_target = "output_data_preprocess_aicpu_"

    def __init__(self, **kwargs):
        # get device_id and device_target
        self._get_devid_and_devtarget()
        self._get_output_path(kwargs)

        os.environ['PROFILING_MODE'] = 'true'
        os.environ['MINDDATA_PROFILING_DIR'] = self._output_path

        if self._device_target:
            CPUProfiler = c_expression.CPUProfiler
            self._cpu_profiler = CPUProfiler.get_instance()
            self._cpu_profiler.init(self._output_path)
            self._cpu_profiler.step_profiling_enable(True)
        if self._device_target and self._device_target == "GPU":
            GPUProfiler = c_expression.GPUProfiler
            self._gpu_profiler = GPUProfiler.get_instance()
            self._gpu_profiler.init(self._output_path)
            self._gpu_profiler.step_profiling_enable(True)
            if GlobalComm.WORLD_COMM_GROUP == "nccl_world_group":
                self._dev_id = str(get_rank())
            os.environ['DEVICE_ID'] = self._dev_id

            if kwargs:
                logger.warning("Params not be supported yet on GPU.")
        elif self._device_target and self._device_target == "Ascend":
            optypes_not_deal = kwargs.pop("optypes_not_deal", "Variable")
            if not isinstance(optypes_not_deal, str):
                raise TypeError("The parameter optypes_not_deal must be str.")
            job_dir = kwargs.pop("ascend_job_id", "")
            if job_dir:
                job_dir = validate_and_normalize_path(job_dir)
                if not os.path.exists(job_dir):
                    msg = f"Invalid ascend_job_id: {job_dir}, Please pass the absolute path of the JOB dir"
                    logger.error(msg)
                    raise ValueError(msg)
                self._output_path, _ = os.path.split(job_dir)
            if kwargs:
                logger.warning("There are invalid params which don't work.")

            os.environ['DEVICE_ID'] = self._dev_id
            fp_point = os.environ.get("PROFILING_FP_START", "")
            bp_point = os.environ.get("PROFILING_BP_END", "")

            profiling_options = {
                "output": self._output_path,
                "fp_point": fp_point,
                "bp_point": bp_point,
                "training_trace": "on",
                "task_trace": "on",
                "aic_metrics": "PipeUtilization",
                "aicpu": "on"
            }

            profiling_options = json.dumps(profiling_options)
            # Characters longer than 2048 are ignored, resulting in profiling option resolution errors
            if len(profiling_options) > 2048:
                msg = "The parameter length exceeds the limit (2048), please input valid parameters."
                logger.error(msg)
                raise ValueError(msg)
            # use context interface to open profiling, for the new mindspore version(after 2020.5.21)
            context.set_context(enable_profiling=True, profiling_options=profiling_options)
            base_profiling_container_path = os.path.join(self._output_path, "container")
            container_path = os.path.join(base_profiling_container_path, self._dev_id)
            data_path = os.path.join(container_path, "data")
            data_path = validate_and_normalize_path(data_path)
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)

            self._filt_optype_names = optypes_not_deal.split(",") if optypes_not_deal else []
            # add job id env through user input later
            self._job_id_env = 0
            self._start_time = int(time.time() * 10000000)
            logger.info("Profiling: profiling start time: %d", self._start_time)

    def analyse(self):
        """
        Collect and analyse performance data, called after training or during training. The example shows above.
        """
        self._cpu_profiler.stop()
        _stop_dataset_profiler()
        if self._device_target and self._device_target == "GPU":
            self._gpu_analyse()

        elif self._device_target and self._device_target == "Ascend":
            self._ascend_analyse()

    def _ascend_analyse(self):
        """Collect and analyse ascend performance data"""
        release()

        job_id = self._get_profiling_job_id()
        logger.info("Profiling: job id is %s ", job_id)

        source_path = os.path.join(self._output_path, job_id)
        # parse hwts.log.data.45.dev file, and get task profiling data
        hwts_output_filename = self._hwts_output_filename_target + self._dev_id + ".txt"
        hwts_output_filename = os.path.join(self._output_path, hwts_output_filename)
        source_path = validate_and_normalize_path(source_path)
        hwts_output_filename = validate_and_normalize_path(hwts_output_filename)
        hwtslog_parser = HWTSLogParser(source_path, hwts_output_filename)
        hwtslog_parser.execute()

        # parse Framework file, and get the relation of op and tasks
        framework_parser = FrameworkParser(job_id, self._dev_id, self._output_path)
        framework_parser.parse()
        op_task_dict = framework_parser.to_task_id_full_op_name_dict()
        if not op_task_dict:
            logger.error("Profiling: fail to parse framework files.")
            return

        # get op compute time from hwts data and framework data, write output_op_compute_time.txt
        opcompute_output_filename = self._opcompute_output_filename_target + self._dev_id + ".txt"
        opcompute_output_filename = os.path.join(self._output_path, opcompute_output_filename)
        opcompute_output_filename = validate_and_normalize_path(opcompute_output_filename)
        optime_parser = OPComputeTimeParser(
            hwts_output_filename, opcompute_output_filename,
            op_task_dict, self._output_path, self._dev_id
        )
        optime_parser.execute()

        # parse DATA_PREPROCESS.dev.AICPU file, write output_data_preprocess_aicpu_x.txt
        output_data_preprocess_aicpu = self._aicpu_op_output_filename_target + self._dev_id + ".txt"
        output_data_preprocess_aicpu = os.path.join(self._output_path, output_data_preprocess_aicpu)
        output_data_preprocess_aicpu = validate_and_normalize_path(output_data_preprocess_aicpu)
        aicpu_data_parser = DataPreProcessParser(source_path, output_data_preprocess_aicpu)
        aicpu_data_parser.execute()

        # Parsing minddata AICPU profiling
        MinddataParser.execute(source_path, self._output_path, self._dev_id)

        # parse minddata pipeline operator and queue
        try:
            pipeline_parser = MinddataPipelineParser(self._output_path, self._dev_id, self._output_path)
            pipeline_parser.parse()
        except ProfilerException as err:
            logger.warning(err.message)

        # analyse op compute time info
        try:
            self._analyser_op_info()
        except ProfilerException as err:
            logger.warning(err.message)

        # analyse step trace info
        points = None
        try:
            points = self._analyse_step_trace(source_path, framework_parser)
        except ProfilerException as err:
            logger.warning(err.message)

        # analyse timeline info
        try:
            self._analyse_timeline(aicpu_data_parser, optime_parser, source_path)
        except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
            logger.warning('Fail to write timeline data: %s', err)

        # analyse memory usage info
        try:
            self._analyse_memory_usage(points)
        except (ProfilerIOException, ProfilerFileNotFoundException, ProfilerRawFileException) as err:
            logger.warning(err.message)

        os.environ['PROFILING_MODE'] = str("false")
        context.set_context(enable_profiling=False)

    def _gpu_analyse(self):
        """Collect and analyse gpu performance data"""
        self._dev_id = context.get_context("device_id")
        if GlobalComm.WORLD_COMM_GROUP == "nccl_world_group":
            self._dev_id = str(get_rank())
        self._gpu_profiler.stop()
        timeline_generator = self._generate_timeline()

        # parse minddata pipeline operator and queue for GPU
        try:
            pipeline_parser = MinddataPipelineParser(self._output_path, self._dev_id, self._output_path)
            pipeline_parser.parse()
        except ProfilerException as err:
            logger.warning(err.message)

        # analyse step trace info
        try:
            self._analyse_step_trace(is_training_mode_flag=timeline_generator.check_op_name('Gradients'))
        except ProfilerException as err:
            logger.warning(err.message)

        os.environ['PROFILING_MODE'] = str("false")

        logger.warning(
            '\nMemory Usage is not supported on GPU currently.\n'
            'Please running on Ascend if you would like to see memory analysis, '
            'otherwise, this warning can be ignored.'
        )

    def _analyse_step_trace(self, source_path=None, framework_parser=None, is_training_mode_flag=True):
        """
        Analyse step trace data and save the result.

        Args:
            source_path (str): The directory that contains the step trace original data.
            framework_parser (FrameworkParser): The framework parse instance.
            is_training_mode_flag (bool): Whether in training mode or not.
        """
        logger.info("Begin to parse step trace.")
        # construct output path
        step_trace_intermediate_file_path = os.path.join(
            self._output_path,
            f'step_trace_raw_{self._dev_id}_detail_time.csv'
        )
        point_info_file_path = os.path.join(
            self._output_path,
            'step_trace_point_info.json'
        )
        step_trace_intermediate_file_path = validate_and_normalize_path(step_trace_intermediate_file_path)
        point_info_file_path = validate_and_normalize_path(point_info_file_path)

        if self._device_target and self._device_target == 'GPU':
            input_file_path = os.path.join(
                self._output_path,
                f'step_trace_profiling_{self._dev_id}.txt'
            )
            parser = GpuStepTraceParser(input_dir=input_file_path,
                                        output_file_path=step_trace_intermediate_file_path,
                                        is_training_mode=is_training_mode_flag)
            parser.parse_and_save()
            point_info = parser.record_point_info(input_file_path, point_info_file_path)
        else:
            # whether keep the first step
            skip_first_step_flag = framework_parser.check_op_name(INIT_OP_NAME)
            point_info = framework_parser.point_info
            # recognize inference or training mode
            is_traning_mode_flag = framework_parser.check_op_name("Gradients")
            # parser the step trace files and save the result to disk
            source_path = validate_and_normalize_path(source_path)
            parser = AscendStepTraceParser(input_dir=source_path,
                                           output_file_path=step_trace_intermediate_file_path,
                                           job_id=self._job_id_env,
                                           skip_first_step=skip_first_step_flag,
                                           is_training_mode=is_traning_mode_flag)
            parser.update_tag_op_type_map(point_info)
            parser.parse_and_save()
            point_info = parser.record_point_info(point_info, point_info_file_path)
        # print parser result
        parser.show()
        logger.info("Finish saving the intermediate result: %s", step_trace_intermediate_file_path)
        logger.info("The point info is: %s", point_info)

        return point_info

    def _analyse_timeline(self, aicpu_parser, optime_parser, source_path):
        """
        Analyse and parse timeline info.

        Args:
            aicpu_parser (DataPreProcessParser): The parser instance for AI CPU operator
                execution time calculation.
            optime_parser (OPComputeTimeParserParser): The parser instance for AI Core
                operator execution time calculation.
        """
        timeline_analyser = AscendTimelineGenerator(self._output_path, self._dev_id)
        # Get framework info
        integrator = Integrator(self._output_path, self._dev_id)
        aicore_detail_data = integrator.get_aicore_detail_data()
        aicore_detail_data_size = len(aicore_detail_data)
        col_names = ['op_name', 'op_type', 'avg_execution_time', 'subgraph',
                     'full_op_name', 'op_info']
        framework_info = {
            'col_name': col_names,
            'object': aicore_detail_data,
            'size': aicore_detail_data_size
        }

        all_reduce_info = integrator.query_for_all_reduce()

        # Get timeline info
        logger.info('Start writing timeline info...')
        logger.info('Warm Prompt: It could take a few minutes if you are training '
                    'with a complex network or more than 10 steps.')
        # Add info into timeline, such as AI CPU, AllReduce, framework info.
        aicpu_info = aicpu_parser.query_aicpu_data()
        min_cycle_counter = min(aicpu_parser.min_cycle_counter, optime_parser.min_cycle_counter)
        timeline_analyser.init_timeline(all_reduce_info, framework_info, aicpu_info,
                                        min_cycle_counter, source_path)
        size_limit = 20 * 1024 * 1024  # 20MB
        timeline_analyser.write_timeline(size_limit)
        timeline_analyser.write_timeline_summary()

    def _generate_timeline(self):
        """Used for gpu, generate timeline info, write to json format file."""
        try:
            size_limit = 100 * 1024 * 1024  # 100MB
            timeline_generator = GpuTimelineGenerator(self._output_path, self._dev_id)
            timeline_generator.init_timeline()
            timeline_generator.write_timeline(size_limit)
            timeline_generator.write_timeline_summary()
            return timeline_generator
        except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
            logger.warning('Fail to write timeline data: %s', err)
            raise RuntimeError('Fail to write timeline data.')

    def _analyse_memory_usage(self, points):
        """Analyse memory usage data."""
        integrator = Integrator(self._output_path, self._dev_id)
        aicore_detail_data = integrator.get_aicore_detail_data()
        memory_parser = MemoryUsageParser(self._output_path, self._dev_id)
        memory_parser.init_memory_usage_info(aicore_detail_data, points)
        memory_parser.write_memory_files()

    def _get_profiling_job_id(self):
        """Get profiling job id, which was generated by ada service.

        Returns:
            str, profiling job id.
        """

        job_id = ""

        for item in os.listdir(self._output_path):
            if item.startswith('JOB'):
                path = os.path.join(self._output_path, item)

                log_file = get_file_names(path, "host_start.log")
                if not log_file:
                    logger.error("Profiling: job path %s, host_start.log not exist.", path)
                    continue

                training_device_id = log_file[0].split('.')[-1]
                if self._dev_id == training_device_id:
                    log_file = os.path.join(path, log_file[0])
                    job_start_time = self._parse_host_start_log(log_file)
                    if not job_start_time:
                        logger.error("Profiling: job path %s, fail to get job start info.", path)
                        break
                    job_id = item
                    if self._start_time > int(job_start_time):
                        logger.info("Profiling: job path %s, start_time %s, training start_time %d.",
                                    path, job_start_time, self._start_time)
                    break
                else:
                    logger.info("Profiling: job path %s, dev id %s, training device id %s.",
                                path, training_device_id, self._dev_id)

        if not job_id:
            msg = "Fail to get profiling job, please check whether job dir was generated"
            raise RuntimeError(msg)

        return job_id

    def _parse_host_start_log(self, input_file):
        """
        Parse host start log file, get the start time of the job.

        Args:
             input_file (str): The file path of the host start log file.

        Returns:
            str, job start time.
        """

        job_start_time = ""
        with open(input_file) as f:
            for line in f.readlines():
                if "clock_realtime" in line:
                    # 16 means the first digit of the timestamp, len(line)-3 means the last.
                    job_start_time = line[16:len(line)-3]

        return job_start_time

    def _analyser_op_info(self):
        """Analyse the operator information."""
        integrator = Integrator(self._output_path, self._dev_id)
        integrator.integrate()

        aicore_type_result = self._query_op_type_info()
        detail_file_path = os.path.join(
            self._output_path,
            'output_op_compute_time_detail_{}.txt'.format(self._dev_id)
        )
        fwrite_format(detail_file_path, data_source='title:op compute time')
        display_names = [
            'optype_name', 'compute_time(ms, per-step)',
            'called_times(per-step)', 'percent'
        ]
        fwrite_format(detail_file_path, data_source=" ".join(display_names), is_print=True)
        fwrite_format(detail_file_path, data_source=aicore_type_result, is_print=True)

        op_type_order = [item[0] for item in aicore_type_result]
        aicore_detail_result = self._query_op_detail_info(op_type_order)

        fwrite_format(detail_file_path, data_source='', is_print=True)
        fwrite_format(detail_file_path, data_source='Detail:', is_print=True)
        fwrite_format(detail_file_path, data_source=" ".join(aicore_detail_result.get('col_name_detail')),
                      is_print=True)
        fwrite_format(detail_file_path, data_source=aicore_detail_result.get('object'), is_print=True)

    def _query_op_type_info(self):
        """
        Query AICORE operator type information.

        Returns:
            list[list], the AICORE operator type and execution time information.
        """
        integrator = Integrator(self._output_path, self._dev_id)
        return integrator.get_aicore_data()

    def _query_op_detail_info(self, op_type_order):
        """
        Query AICORE operator detail information.

        Args:
            op_type_order(list): The name of the op type in order.

        Returns:
            dict, the AICORE operator detail information.
        """

        op_type_condition = {}
        if self._filt_optype_names:
            op_type_condition['not_in'] = self._filt_optype_names

        filter_condition = {
            'op_type': op_type_condition,
            'is_display_detail': False,
        }
        integrator = Integrator(self._output_path, self._dev_id)
        return integrator.query_and_sort_by_op_type(filter_condition, op_type_order)

    def _get_devid_and_devtarget(self):
        """Get device id and target of this training."""

        device_target = ""
        dev_id = ""
        try:
            dev_id = str(context.get_context("device_id"))
            device_target = context.get_context("device_target")
        except ValueError as err:
            logger.error("Profiling: fail to get context, %s", err)

        if not dev_id or not dev_id.isdigit():
            dev_id = os.getenv('DEVICE_ID')
        if not dev_id or not dev_id.isdigit():
            dev_id = "0"
            logger.error("Fail to get DEVICE_ID, use 0 instead.")

        if device_target and device_target not in ["Ascend", "GPU"]:
            msg = "Profiling: unsupported backend: %s" % device_target
            raise RuntimeError(msg)

        self._dev_id = dev_id
        self._device_target = device_target

    def _get_output_path(self, kwargs):
        """Get output path of profiling data."""
        current_time = int(time.time())

        # to avoid getting different timestamp from different process in multi-card training,
        # set the timestamp as exist timestamp if it's difference is less than 6 seconds.
        def _select_timestamp(dir_name, re_pattern, input_time):
            """select the timestamp from current_time and exist timestamp."""
            timestamp_diff_threshold = 6
            exist_timestamp_list = []
            select_time = input_time
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            for file_name in os.listdir(dir_name):
                match_res = re_pattern.match(file_name)
                if match_res:
                    exist_timestamp_list.append(int(match_res.group(1)))
            if exist_timestamp_list:
                time_diff_list = [input_time - timestamp for timestamp in exist_timestamp_list]
                min_time_diff = min(time_diff_list)
                if min_time_diff <= timestamp_diff_threshold:
                    select_time = exist_timestamp_list[time_diff_list.index(min_time_diff)]

            return select_time

        if "output_path" not in kwargs:
            selected_timestamp = _select_timestamp(os.getcwd(), re.compile(r'data-(\d+)'), current_time)
            output_path = f"data-{selected_timestamp}"
            self._output_path = validate_and_normalize_path(output_path)
        else:
            output_path = kwargs.pop("output_path")
            self._output_path = validate_and_normalize_path(output_path)
            selected_timestamp = _select_timestamp(self._output_path,
                                                   re.compile(r'profiler-(\d+)'), current_time)

        self._output_path = os.path.join(self._output_path, f"profiler-{selected_timestamp}")
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path, exist_ok=True)
            os.chmod(self._output_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        else:
            logger.warning("The target dir already exists. "
                           "There may be some old profiling data, and they will be rewrote in the end.")

    @staticmethod
    def profile(network=None, profile_option=None):
        """
        Get the number of trainable parameters in the training network.

        Args:
            network (Cell): The training network.
            profile_option (ProfileOption): The profile option.

        Returns:
            dict, the key is the option name, the value is the result of option.
        """
        result = dict()
        if not profile_option:
            raise ValueError("The parameter profile_option must pass a value using ProfileOption.")

        if profile_option == ProfileOption.trainable_parameters:
            if not isinstance(network, Cell):
                msg = "Profiling: The network should be an object of nn.Cell"
                raise ValueError(msg)
            param_nums = len(network.parameters_dict())
            result = {"trainable_parameters": param_nums}
        else:
            raise ValueError("Wrong options.")

        return result
