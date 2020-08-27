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
"""Profiling api file."""
import os
import time

from mindspore import log as logger, context
from mindspore.communication.management import release
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException, \
    ProfilerIOException, ProfilerException
from mindspore.profiler.common.util import get_file_names, fwrite_format
from mindspore.profiler.common.validator.checkparam import \
    check_bool, check_subgraph
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path
from mindspore.profiler.parser.aicpu_data_parser import DataPreProcessParser
from mindspore.profiler.parser.framework_parser import FrameworkParser
from mindspore.profiler.parser.hwts_log_parser import HWTSLogParser
from mindspore.profiler.parser.integrator import Integrator
from mindspore.profiler.parser.integrator import TimelineAnalyser
from mindspore.profiler.parser.minddata_parser import MinddataParser
from mindspore.profiler.parser.minddata_pipeline_parser import \
    MinddataPipelineParser
from mindspore.profiler.parser.optime_parser import OPComputeTimeParser
from mindspore.profiler.parser.step_trace_parser import StepTraceParser
from mindspore.nn.cell import Cell

PROFILING_LOG_BASE_PATH = "/var/log/npu/profiling"
INIT_OP_NAME = 'Default/InitDataSetQueue'


class Profiler:
    """
    Performance profiling API.

    Enable MindSpore users to profile the performance of neural network.
    Profiler support Ascend and GPU, both of them are used in the same way,
    but only output_path in args works on GPU.

    Args:
        subgraph (str): (Ascend only)Define which subgraph to monitor and analyse, can be 'all', 'Default', 'Gradients'.
        is_detail (bool): (Ascend only)Whether to show profiling data for op_instance level,
            only show optype level if False.
        is_show_op_path (bool): (Ascend only)Whether to save the full path for each op instance.
        output_path (str): Output data path.
        optypes_to_deal (str): (Ascend only)Op type names, the data of which optype should be collected and analysed,
            will deal with all op if null; Different op types should be seperated by comma.
        optypes_not_deal (str): (Ascend only)Op type names, the data of which optype will not be collected and analysed;
            Different op types should be seperated by comma.
        job_id (str): (Ascend only)The directory where the parsed profiling files are located;
            This parameter is used to support offline parsing.

    Examples:
        >>> from mindspore.profiler import Profiler
        >>> import mindspore.context
        >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
        >>>                     device_id=int(os.environ["DEVICE_ID"]))
        >>> profiler = Profiler()
        >>> model = Model()
        >>> model.train()
        >>> profiler.analyse()
    """

    _base_profiling_container_path = "/var/log/npu/profiling/container"
    _hwts_output_filename_target = "output_format_data_hwts_"
    _opcompute_output_filename_target = "output_op_compute_time_"
    _aicpu_op_output_filename_target = "output_data_preprocess_aicpu_"

    def __init__(self, subgraph='all', is_detail=True, is_show_op_path=False, output_path='./data',
                 optypes_to_deal='', optypes_not_deal='Variable', job_id=""):
        # get device_id and device_target
        self._get_devid_and_devtarget()
        self._output_path = validate_and_normalize_path(output_path)
        self._output_path = os.path.join(self._output_path, "profiler")
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path, exist_ok=True)
        else:
            logger.warning("The target dir already exists. "
                           "There may be some old profiling data, and they will be rewrote in the end.")

        if self._device_target and self._device_target == "GPU":
            from mindspore._c_expression import GPUProfiler
            self._gpu_profiler = GPUProfiler.get_instance()
            self._gpu_profiler.init(self._output_path)
            self._gpu_profiler.step_profiling_enable(True)
        elif self._device_target and (self._device_target == "Ascend" or self._device_target != "Davinci"):
            self._container_path = os.path.join(self._base_profiling_container_path, self._dev_id)
            data_path = os.path.join(self._container_path, "data")
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)

            os.environ['PROFILING_MODE'] = 'true'
            os.environ['PROFILING_OPTIONS'] = 'training_trace:task_trace'
            os.environ['MINDDATA_PROFILING_DIR'] = self._output_path
            os.environ['DEVICE_ID'] = self._dev_id
            os.environ['AICPU_PROFILING_MODE'] = 'true'
            os.environ['PROFILING_DIR'] = str(self._container_path)

            # use context interface to open profiling, for the new mindspore version(after 2020.5.21)
            context.set_context(enable_profiling=True, profiling_options="training_trace:task_trace")

            self._subgraph = check_subgraph(subgraph)
            self._valid_optype_name = optypes_to_deal.split(",") if optypes_to_deal else []
            self._filt_optype_names = optypes_not_deal.split(",") if optypes_not_deal else []
            self._detail = check_bool(is_detail, 'is_detail')
            self._withfullpath = check_bool(is_show_op_path, 'is_show_op_path')
            self._profiling_job_id = job_id
            # add job id env through user input later
            self._job_id_env = 0
            self._start_time = int(time.time() * 10000000)
            logger.info("Profiling: profiling start time: %d", self._start_time)

    def analyse(self):
        """
        Collect and analyse performance data, called after training or during training.

        Examples:
            >>> from mindspore.profiler import Profiler
            >>> import mindspore.context
            >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
            >>>                     device_id=int(os.environ["DEVICE_ID"]))
            >>> profiler = Profiler(subgraph='all', is_detail=True, is_show_op_path=False, output_path='./data')
            >>> model = Model()
            >>> model.train()
            >>> profiler.analyse()
        """
        if self._device_target and self._device_target == "GPU":
            self._gpu_profiler.stop()
        elif self._device_target and (self._device_target == "Ascend" or self._device_target != "Davinci"):
            release()

            job_id = self._get_profiling_job_id()
            logger.info("Profiling: job id is %s ", job_id)

            source_path = os.path.join(PROFILING_LOG_BASE_PATH, job_id)
            # parse hwts.log.data.45.dev file, and get task profiling data
            hwts_output_filename = self._hwts_output_filename_target + self._dev_id + ".txt"
            hwts_output_filename = os.path.join(self._output_path, hwts_output_filename)
            hwtslog_parser = HWTSLogParser(source_path, hwts_output_filename)
            result = hwtslog_parser.execute()
            if not result:
                logger.error("Profiling: fail to parse hwts log file.")
                return

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
            optime_parser = OPComputeTimeParser(
                hwts_output_filename, opcompute_output_filename,
                op_task_dict, self._output_path, self._dev_id
            )
            optime_parser.execute()

            # parse DATA_PREPROCESS.dev.AICPU file, write output_data_preprocess_aicpu_x.txt
            output_data_preprocess_aicpu = self._aicpu_op_output_filename_target + self._dev_id + ".txt"
            output_data_preprocess_aicpu = os.path.join(self._output_path, output_data_preprocess_aicpu)
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
            try:
                self._analyse_step_trace(source_path, framework_parser)
            except ProfilerException as err:
                logger.warning(err.message)

            # analyse timeline info
            try:
                self._analyse_timeline(aicpu_data_parser, optime_parser)
            except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
                logger.warning('Fail to write timeline data: %s', err)

            os.environ['PROFILING_MODE'] = str("false")
            context.set_context(enable_profiling=False)

    def _analyse_step_trace(self, source_path, framework_parser):
        """
        Analyse step trace data and save the result.

        Args:
            source_path (str): The directory that contains the step trace original data.
            framework_parser (FrameworkParser): The framework parse instance.
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
        # whether keep the first step
        skip_first_step_flag = framework_parser.check_op_name(INIT_OP_NAME)
        point_info = framework_parser.point_info
        # parser the step trace files and save the result to disk
        parser = StepTraceParser(input_dir=source_path,
                                 output_file_path=step_trace_intermediate_file_path,
                                 job_id=self._job_id_env,
                                 skip_first_step=skip_first_step_flag)
        parser.update_tag_op_type_map(point_info)
        parser.parse_and_save()
        point_info = parser.record_point_info(point_info, point_info_file_path)
        # print parser result
        parser.show()
        logger.info("Finish saving the intermediate result: %s", step_trace_intermediate_file_path)
        logger.info("The point info is: %s", point_info)

    def _analyse_timeline(self, aicpu_parser, optime_parser):
        """
        Analyse and parse timeline info.

        Args:
            aicpu_parser (DataPreProcessParser): The parser instance for AI CPU operator
                execution time calculation.
            optime_parser (OPComputeTimeParserParser): The parser instance for AI Core
                operator execution time calculation.
        """
        timeline_analyser = TimelineAnalyser(self._output_path, self._dev_id)
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
        timeline_analyser.init_timeline(all_reduce_info, framework_info, aicpu_info, min_cycle_counter)
        timeline_analyser.write_timeline()
        timeline_analyser.write_timeline_summary()

    def _get_profiling_job_id(self):
        """Get profiling job id, which was generated by ada service.

        Returns:
            str: profiling jon id.
        """

        if self._profiling_job_id:
            return self._profiling_job_id

        job_id = ""
        cmd = "ls -t " + PROFILING_LOG_BASE_PATH + "|grep JOB|awk '{print $1}'"
        r = os.popen(cmd)
        profiling_job_dirs = r.readlines()
        r.close()
        for item in profiling_job_dirs:
            path = os.path.join(PROFILING_LOG_BASE_PATH, item.strip())
            log_file = get_file_names(path, "host_start.log")
            if not log_file:
                logger.error("Profiling: job path %s, host_start.log not exist.", path)
                continue

            log_file = os.path.join(path, log_file[0])
            item_dict = self._parse_host_start_log(log_file)

            if not item_dict:
                logger.error("Profiling: job path %s, fail to get job start info.", path)
                continue
            if self._start_time > int(item_dict["start_time"]):
                logger.info("Profiling: job path %s, start_time %s, training start_time %d.",
                            path, item_dict["start_time"], self._start_time)
                break

            if self._dev_id != item_dict["device_id"]:
                logger.info("Profiling: job path %s, dev id %s, training device id %s.",
                            path, item_dict["device_id"], self._dev_id)
                continue

            job_id = item.strip()
            break

        if not job_id:
            msg = "Fail to get profiling job, please check whether job dir was generated"
            raise RuntimeError(msg)

        return job_id

    def _parse_host_start_log(self, input_file):
        """
        Parse host start log file, get the device id and start time of the job.

        Args:
             input_file (str): The file path of the host start log file.

        Returns:
            dict, job start time and device id.
        """

        item_dict = {}
        for line in open(input_file):
            if "Device" in line:
                item_dict["device_id"] = line[7:len(line)-2]
            elif "clock_realtime" in line:
                item_dict["start_time"] = line[16:len(line)-3]

        return item_dict

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

        if self._detail:
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
        if self._valid_optype_name:
            op_type_condition['in'] = self._valid_optype_name
        if self._filt_optype_names:
            op_type_condition['not_in'] = self._filt_optype_names

        subgraph_condition = {}
        if self._subgraph != 'all':
            subgraph_condition['in'] = [self._subgraph]

        filter_condition = {
            'op_type': op_type_condition,
            'subgraph': subgraph_condition,
            'is_display_detail': False,
            'is_display_full_op_name': self._withfullpath
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

        if device_target and device_target not in ["Davinci", "Ascend", "GPU"]:
            msg = "Profiling: unsupported backend: %s" % device_target
            raise RuntimeError(msg)

        self._dev_id = dev_id
        self._device_target = device_target

    @staticmethod
    def trainable_parameters(network):
        """
        Get the number of trainable parameters in the training network.

        Args:
            network(Cell): The training network.

        Returns:
            an integer,the network of trainable parameters.
        """
        if not isinstance(network, Cell):
            msg = "Profiling: The network should be an object of nn.Cell"
            raise ValueError(msg)

        param_nums = len(network.parameters_dict())

        return param_nums
