# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Checkpoint related classes and functions."""

import time
import math
import json
import os
import textwrap
from mindspore.utils import _tft_handler
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback._callback import Callback
from mindspore.train.callback._checkpoint import ModelCheckpoint
from mindspore import context, ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore.communication.management import GlobalComm
from mindspore import log as logger
from mindspore.parallel._cell_wrapper import SingleCommunicator
from mindspore.train.serialization import _get_cur_rank_dp
from mindspore._c_expression import _repair_device, _stop_device, _tft_sem_post, _tft_sem_enable
from mindspore._c_expression import _rebuild_group, _finalize_comm
from mindspore._c_expression import clean_tdt_channel
from mindspore._c_expression import _pre_launch_send_recv
from mindspore._c_expression import send_recv, reset_params, direct_copy_to_host
from mindspore._c_expression import _reg_snapshot_params, _reset_snapshot_state, _clear_snapshot_saving_flag
from mindspore._c_expression import CollectiveManager
from mindspore._c_expression import _get_uce_process_strategy, _get_uce_mem_info, _reset_opt_event_info
from mindspore.ops.operations.manually_defined._inner import TensorReport
import mindspore
import mindspore.common.dtype as mstype
from mindspore import runtime
from mindspore._c_expression import set_is_arf, check_is_arf


def _get_ckpt_dir(step, ckpt_save_path, is_tmp_file):
    """ Common func to generate ckpt dir name."""
    tmp = "_tmp" if is_tmp_file else ""
    mid_dir = f"ttp_saved_checkpoints-step_{str(step)}{tmp}"
    return os.path.join(ckpt_save_path, mid_dir)


def _save_checkpoint_on_failure(step, save_info, args, cb_ctx):
    """ Callback used for TFT save ckpt function when errors occur."""
    logger.info("Enter _save_checkpoint_on_failure function")
    if not cb_ctx._is_params_consistent():  # pylint: disable=W0212
        raise RuntimeError("Can't save parameters, because they are left in inconsistent state!")
    cb_params = args
    # we record the current step and epoch num in on_train_step_end, so we can just reset it here
    cb_params.cur_step_num = cb_ctx.cur_step_num
    cb_params.cur_epoch_num = cb_ctx.cur_epoch_num
    if cb_params.optimizer is not None:
        cb_params.optimizer.global_step = cb_ctx.global_step
    if hasattr(cb_params.network, 'optimizer') and cb_params.network.optimizer is not None:
        cb_params.network.optimizer.global_step = cb_ctx.global_step
    append_dict = {}
    append_dict["__exception_save__"] = True
    # if user has provided a custom save callback, use it
    if cb_ctx.save_cb:
        cb_ctx.save_cb(cb_params, append_dict)
        logger.info("Finish _save_checkpoint_on_failure function")
        return

    # if user has not provided a custom save callback, use default save logic
    ckpt_save_path = cb_ctx.ckpt_save_path
    cur_rank = get_rank()
    step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
    cur_epoch_num = cb_params.cur_epoch_num
    append_dict["epoch_num"] = cur_epoch_num
    append_dict["step_num"] = cb_params.cur_step_num
    append_dict["cur_rank"] = cur_rank
    append_dict["batch_num"] = cb_params.batch_num
    append_dict["global_step"] = cb_ctx.global_step
    outputs = cb_params.net_outputs
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        append_dict["loss_scale"] = outputs[2]

    ckpt_file = f"ttp_rank_{str(cur_rank)}-{str(cur_epoch_num)}_{str(step_num_in_epoch)}.ckpt"
    cur_ckpt_dir = os.path.join(_get_ckpt_dir(step, ckpt_save_path, True), "rank_" + str(cur_rank))
    os.makedirs(cur_ckpt_dir, exist_ok=True)
    cur_file = os.path.join(cur_ckpt_dir, ckpt_file)
    save_checkpoint(cb_params.train_network, cur_file,
                    integrated_save=False, append_dict=append_dict)
    logger.info("Finish _save_checkpoint_on_failure function")


def _rename_save_result(step, cb_ctx):
    """ Callback used for TFT rename function after ckpt save callback was finished and successful."""
    logger.info("Enter _rename_save_result function")
    if cb_ctx.save_cb:
        logger.info("User's save callback is provided, skip rename")
        return
    tmp_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, True)
    fin_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, False)

    os.rename(tmp_dir, fin_dir)
    logger.info("Finish _rename_save_result function")


def _tft_exit_cb(ctx):
    """Callback used for TFT exit function."""
    logger.error("Enter mindio ttp exit process, which means other ranks occur exception, check other ranks' logs!")
    _tft_sem_post()
    os._exit(1)  # pylint: disable=W0212


def _tft_repair_callback(step, need_rebuild, error_ranks, repair_info, args, cb_ctx):
    """ Callback used for TFT repair function."""
    logger.warning(f"Enter _tft_repair_callback repair type: {repair_info['repair_type']}")
    if (repair_info["repair_type"] in (cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value,
                                       cb_ctx.tft.RepairType.RT_UCE_LOWLEVEL.value)):
        logger.warning("Enter _tft_repair_callback uce REPARI_DEVICE device_id : {}".format(cb_ctx.device_id))
        _repair_device(cb_ctx.device_id)

    if (repair_info["repair_type"] in (cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value,
                                       cb_ctx.tft.RepairType.RT_SEND.value,
                                       cb_ctx.tft.RepairType.RT_RECV_REPAIR.value)):
        logger.warning("Enter _tft_repair_callback SEND_RECV repair type:{}, src_rank:{}, dst_rank: {}".format(
            repair_info["repair_type"], repair_info["src"], repair_info["dst"]))
        cb_params = args
        if repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_SEND.value:
            for i in range(len(repair_info["src"])):
                src_rank = repair_info["src"][i]
                dst_rank = repair_info["dst"][i]
                if send_recv(cb_params.train_network.trainable_params(), src_rank, dst_rank) != 0:
                    raise ValueError("Call send_recv failed.")
        else:
            src_rank = repair_info["src"][0]
            dst_rank = repair_info["dst"][0]
            if send_recv(cb_params.train_network.trainable_params(), src_rank, dst_rank) != 0:
                raise ValueError("Call send_recv failed.")
    logger.warning("Finish _tft_repair_callback")


def _tft_clean_callback(is_uce_error, args, ctx):
    """ Callback used for TFT clean function."""
    logger.warning(f"Enter _tft_clean_callback, device id:{ctx.device_id}")
    ret = 0
    if is_uce_error:
        _get_uce_mem_info(ctx.device_id)
        err_strategy = _get_uce_process_strategy()
        logger.warning("_tft_clean_callback err_strategy: {}".format(err_strategy))
        if err_strategy == "RS_UCE_HIGHLEVEL":
            ret = 0
        elif err_strategy == "RS_UCE_LOWLEVEL":
            ret = 2
        else:
            ret = 1
    clean_tdt_channel()
    if ctx.tft.tft_get_repair_type() == "retry":
        logger.warning("Enter _tft_clean_callback resume_hccl_comm")
        CollectiveManager.get_instance().resume_hccl_comm()
    logger.warning("Finish _tft_clean_callback, ret: {}".format(ret))
    if ctx.tft.tft_get_repair_type() == "recover":
        _reset_snapshot_state()
    return ret


def _tft_stop_callback(args, cb_ctx):
    """ Callback used for TFT stop function."""
    logger.warning(f"Enter _tft_stop_callback device_id: {cb_ctx.device_id}")
    if (not cb_ctx.is_uce_rank) and (not cb_ctx._is_params_consistent()):  # pylint: disable=W0212
        raise RuntimeError("Can't stop device, because training parameters are left in inconsistent state!")
    cb_ctx.is_uce_rank = False
    _stop_device(cb_ctx.device_id)
    if cb_ctx.tft.tft_get_repair_type() == "recover":
        logger.warning("Reset limit step")
        cb_ctx.tft.tft_reset_limit_step()
    logger.warning("Finish _tft_stop_callback")


def _tft_rebuild_groups(fault_ranks, args, ctx):
    """Callback used for TFT Rebuild Group function."""
    logger.warning(f"Enter _tft_rebuild_groups, device id: {ctx.device_id}")
    _finalize_comm()
    _rebuild_group()
    set_is_arf(True)
    logger.warning("try to pre launch send recv before real launch")
    _pre_launch_send_recv(context.get_context('device_id'))
    logger.warning("Pre launch send recv before real launch end")
    logger.warning("Enter _tft_rebuild_groups ok ")


class AdaptTrainOptimizer():
    """
    Adaptive Training checkpoint optimizer for TFT fault handling.

    This class records failure statistics and step/ckpt timing to compute a recommended
    checkpoint interval based on a failure-rate-based policy. Before reaching the ATO
    threshold, it only records statistics and reports insufficient statistics without
    generating a recommendation.

    Note:
        This class does not save checkpoints by itself. It only collects statistics and
        computes a recommended interval.

    Args:
        ato_path (str): Path to the JSON file used to persist ATO statistics.
        ato_threshold (int): Minimum failure count required before applying the
            failure-rate-based policy.
    """
    def __init__(self, ato_path, ato_threshold):
        self._ato_path = ato_path
        self._ato_threshold = ato_threshold
        self._load_failure_rate_from_json(self._ato_path)

        self._interval_comm = SingleCommunicator(GlobalComm.WORLD_COMM_GROUP)
        self._group_size = get_group_size()
        self.step_times = []
        self.checkpoint_times = []

    def bind_ckpt_obj(self, ckpt_obj):
        """Bind the ModelCheckpoint instance for ATO checkpoint logic."""
        if not isinstance(ckpt_obj, ModelCheckpoint):
            raise ValueError("ATO checkpoint optimization expects a ModelCheckpoint instance.")
        self.ckpt_obj = ckpt_obj

    def _load_failure_rate_from_json(self, path):
        """Load failure statistics from a JSON file."""
        if not os.path.exists(path):
            self._failure_count = 0
            self._total_steps = 0
            self._failure_rate = 0
        else:
            with open(path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self._failure_count = info.get("failure_count")
            self._total_steps = info.get("total_steps", 0)
            self._failure_rate = self._failure_count / self._total_steps if self._total_steps else 0.0

    def _save_fault_info_json(self, failure_count, total_steps):
        """Persist failure statistics to a JSON file."""
        if get_rank() != 0:
            return
        data = {
            "failure_count": failure_count,
            "total_steps": total_steps
        }
        ato_dir = os.path.dirname(self._ato_path)
        if ato_dir:
            os.makedirs(ato_dir, exist_ok=True)
        with open(self._ato_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def handle_fault(self, exc, cur_step):
        """
        Record fault info and compute a recommended checkpoint interval.

        This function updates failure statistics, persists them to JSON, and logs a
        recommended save interval when enough timing data is available.
        """
        if isinstance(exc, RuntimeError):
            self._failure_count += 1
        save_checkpoint_steps = self.ckpt_obj._config.save_checkpoint_steps #pylint: disable=W0212
        self._total_steps += cur_step
        self._save_fault_info_json(self._failure_count, self._total_steps)
        self._collect_checkpoint_times()

        self._avg_checkpoint_time = self._get_avg_checkpoint_time()
        self._avg_step_time = self._get_avg_step_time()
        if self._avg_checkpoint_time is None or self._avg_step_time is None:
            logger.warning(f"Insufficient timing data: only {cur_step} steps recorded; "
                           f"checkpoint interval is {save_checkpoint_steps}.")
        else:
            if self._failure_count < self._ato_threshold:
                logger.warning("Insufficient statistics for ATO checkpoint calculation. "
                    f"Current failure_count: {self._failure_count}, ato_threshold: "
                    f"{self._ato_threshold}")
            else:
                self._failure_rate = self._failure_count / self._total_steps if self._total_steps else 0.0
                if self._failure_rate == 0:
                    logger.warning("Failure rate is zero, cannot compute failure-rate-based checkpoint interval.")
                else:
                    new_interval = self._calculate_failure_rate_phase_interval()
                    old_interval_ratio = self._calculate_effective_training_ratio(save_checkpoint_steps)
                    new_interval_ratio = self._calculate_effective_training_ratio(new_interval)
                    self._generate_ckpt_suggestion_log(save_checkpoint_steps, new_interval,
                                                       old_interval_ratio, new_interval_ratio)

    def _calculate_failure_rate_phase_interval(self):
        """Compute recommended interval in failure-rate-based phase."""
        optimal_interval = math.sqrt(2 * self._avg_checkpoint_time / (self._failure_rate * self._avg_step_time))
        new_interval = self._comm_interval_step(optimal_interval)
        return new_interval

    def _calculate_effective_training_ratio(self, interval):
        """Calculate effective training ratio for a checkpoint interval."""
        failure_recovery_time = 0.5 * self._failure_rate * interval * self._avg_step_time
        all_train_time = self._avg_step_time + self._avg_checkpoint_time / interval + failure_recovery_time
        effective_ratio = self._avg_step_time / all_train_time
        return effective_ratio

    def _get_avg_checkpoint_time(self):
        """Get the average checkpoint save time."""
        if self.checkpoint_times:
            return sum(self.checkpoint_times) / len(self.checkpoint_times)
        return None

    def _collect_checkpoint_times(self):
        """Collect completed checkpoint save durations from the bound checkpoint callback."""
        if not hasattr(self.ckpt_obj, "save_info_list"):
            return
        checkpoint_times = []
        for save_info_dict in self.ckpt_obj.save_info_list.values():
            ckpt_info = save_info_dict.get("ckpt", {})
            save_start_time = ckpt_info.get("save_start_time")
            save_end_time = ckpt_info.get("save_end_time")
            if save_start_time is None or save_end_time is None:
                continue
            checkpoint_times.append(max(0.0, save_end_time - save_start_time))
        self.checkpoint_times = checkpoint_times

    def _get_avg_step_time(self, interval_step=2000):
        """Get the average step time."""
        if len(self.step_times) <= 1:
            return None
        if len(self.step_times) <= interval_step:
            return sum(self.step_times[1:]) / len(self.step_times[1:])
        return sum(self.step_times[-interval_step:]) / len(self.step_times[-interval_step:])

    def _comm_interval_step(self, interval_step):
        """Aggregate interval step across ranks and return an integer interval."""
        if self._group_size == 1:
            return max(1, int(round(interval_step)))
        new_interval = self._interval_comm(Tensor(int(round(interval_step)), dtype=mstype.int32))
        return max(1, int(new_interval.asnumpy().item() / self._group_size))

    def _format_duration(self, duration):
        """Format duration values for terminal output."""
        return f"{duration:.3f} s"

    def _format_ratio(self, ratio):
        """Format ratio values as percentages for terminal output."""
        return f"{ratio:.2%}"

    def _build_ckpt_suggestion_table(self, info_items, title="ATO CKPT Recommendations"):
        """Build a formatted table string."""
        section_prefix = "__SECTION__:"

        def max_line_length(text):
            return max(len(line) for line in str(text).splitlines() or [""])

        normal_items = [item for item in info_items if not item[0].startswith(section_prefix)]
        section_titles = [item[0][len(section_prefix):] for item in info_items if item[0].startswith(section_prefix)]
        col1_width = max(len("Item"), max(max_line_length(k) for k, _, _ in normal_items)) + 2
        col2_width = max(len("Value"), max(max_line_length(v) for _, v, _ in normal_items)) + 2
        col3_width = min(78, max(len("Description"), max(max_line_length(d) for _, _, d in normal_items))) + 2
        total_width = col1_width + col2_width + col3_width + 8
        total_width = max(total_width, len(title) + 2)
        if section_titles:
            total_width = max(total_width, max(len(section_title) for section_title in section_titles) + 2)

        lines = []
        lines.append("+" + "-" * total_width + "+")
        lines.append(f"|{title:^{total_width}}|")
        lines.append("+" + "-" * total_width + "+")
        header_line = f"| {'Item':<{col1_width}} | {'Value':<{col2_width}} | {'Description':<{col3_width}} |"

        for key, value, desc in info_items:
            if key.startswith(section_prefix):
                section_title = key[len(section_prefix):]
                lines.append(f"|{section_title:^{total_width}}|")
                lines.append("+" + "-" * total_width + "+")
                lines.append(header_line)
                lines.append("+" + "-" * total_width + "+")
                continue
            key_lines = self._wrap_table_cell(key, col1_width)
            value_lines = self._wrap_table_cell(value, col2_width)
            desc_lines = self._wrap_table_cell(desc, col3_width)
            row_height = max(len(key_lines), len(value_lines), len(desc_lines))
            for idx in range(row_height):
                key_line = key_lines[idx] if idx < len(key_lines) else ""
                value_line = value_lines[idx] if idx < len(value_lines) else ""
                desc_line = desc_lines[idx] if idx < len(desc_lines) else ""
                lines.append(
                    f"| {key_line:<{col1_width}} | {value_line:<{col2_width}} | {desc_line:<{col3_width}} |")
            lines.append("+" + "-" * total_width + "+")
        return "\n".join(lines)

    def _wrap_table_cell(self, text, width):
        """Wrap table cell content while preserving manual newlines."""
        wrapped_lines = []
        content_width = max(1, width - 2)
        for raw_line in str(text).splitlines() or [""]:
            if not raw_line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=content_width,
                    break_long_words=False,
                    break_on_hyphens=False
                ) or [""])
        return wrapped_lines

    def _build_current_status_items(self, old_interval, old_interval_ratio):
        """Build table items for the current status section."""
        return [
            ("__SECTION__:Current Status", "", ""),
            ("Failure Count", str(self._failure_count), "Observed training fault count"),
            ("Total Steps", str(self._total_steps), "Total training steps used for statistics"),
            ("Failure Rate", f"{self._failure_rate:.8f}",
             "Fault count divided by total training steps"),
            ("Step Time", self._format_duration(self._avg_step_time), "Average time per training step"),
            ("CKPT Save Time", self._format_duration(self._avg_checkpoint_time), "Average checkpoint save time"),
            ("Current CKPT Interval", f"{old_interval} steps", "Current checkpoint save interval"),
            ("Current Effective Training Ratio", self._format_ratio(old_interval_ratio),
             "Current pure-training share of the overall end-to-end time"),
        ]

    def _build_effective_training_tuning_items(self, old_interval, new_interval, old_interval_ratio,
                                               new_interval_ratio):
        """Build table items for the tuning suggestion."""
        if old_interval == new_interval:
            return [
                ("__SECTION__:Effective Training Time Tuning Suggestion", "", ""),
                ("Checkpoint Frequency Status", f"{old_interval} steps",
                 "The computed recommendation matches the current checkpoint save interval.\n"
                 "The current setting is already the optimal checkpoint save frequency under "
                 "the observed fault rate and timing statistics.\n"
                 "No yaml update is required."),
            ]

        ckpt_prefix = getattr(self.ckpt_obj, "origin_prefix", None)
        if not isinstance(ckpt_prefix, str) or not ckpt_prefix:
            ckpt_prefix = "example"
        yaml_update_hint = (
            "callbacks:\n"
            "  - type: MFLossMonitor  # Prints training progress information\n"
            "  - type: CheckpointMonitor  # Saves model weights during training\n"
            f"    prefix: \"{ckpt_prefix}\"  # Prefix for saved file names\n"
            f"    save_checkpoint_steps: {new_interval}  # Interval steps for saving model weights"
        )
        value_text = self._format_ratio(new_interval_ratio)
        if old_interval_ratio > 0:
            improvement = (new_interval_ratio - old_interval_ratio) / old_interval_ratio * 100
            value_text = f"{value_text} ({improvement:+.2f}%)"

        return [
            ("__SECTION__:Effective Training Time Tuning Suggestion", "", ""),
            ("Suggested CKPT Interval", f"{new_interval} steps",
             f"Recommended checkpoint save interval derived from the observed fault frequency and timing statistics.\n"
             f"Update the CheckpointMonitor save interval from {old_interval} to {new_interval}.\n"
             "Example yaml configuration:\n"
             f"{yaml_update_hint}\n"
             "Calculation formula: sqrt(2 * t_ckpt / (failure_rate * t_step))"),
            ("Suggested Effective Training Ratio",
             value_text,
             f"Estimated pure-training share of the end-to-end time after applying the suggested checkpoint interval.\n"
             f"The effective training ratio is expected to increase from {self._format_ratio(old_interval_ratio)} "
             f"to {self._format_ratio(new_interval_ratio)}.\n"
             "Calculation formula: (t_step * N) / (t_step * N + t_ckpt * N / I + "
             "0.5 * failure_rate * I * t_step * N).\n"
             "In this formula, I denotes the suggested checkpoint interval."),
        ]

    def _generate_ckpt_suggestion_log(self, old_interval, new_interval, old_interval_ratio, new_interval_ratio):
        """Generate and print ATO checkpoint tuning details."""
        if get_rank() != 0:
            return None
        table_items = self._build_current_status_items(old_interval, old_interval_ratio)
        table_items.extend(self._build_effective_training_tuning_items(
            old_interval, new_interval, old_interval_ratio, new_interval_ratio))
        log_text = f"{self._build_ckpt_suggestion_table(table_items)}\n"
        print(log_text, flush=True)
        return log_text


class TrainFaultTolerance(Callback):
    """
    This callback is used to enable the TFT feature MindIO TFT and will execute TFT operations during training
    process, such as TFT init, report and exception handling.

    Note:
        Required for Ascend graph mode only. And sink size must be less than or equal to 1.

    Args:
        ckpt_save_path (str, optional): Checkpoint save directory when failure occurs. When saved,
            a new directory named 'ttp_saved_checkpoints-step_{cur_step_num}'
            is created in that directory. Default: ``None``.
        kwargs (dict): Other dictionary type parameters. When argument `ckpt_save_path` is ``None``, `kwargs` must
            provide a parameter named `ckpt_save_fn`, which points to a function used to save checkpoint. The
            prototype of `ckpt_save_fn` is ``def save_ckpt(cb_params, append_dict)``. When both `ckpt_save_path`
            and `ckpt_save_fn` are provided, `ckpt_save_fn` is used preferentially.

    Raises:
        Exception: TFT init failed.
        ModuleNotFoundError: Mindio TFT whl package is not installed.

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            It's recommended to use the msrun startup method.
            Please see the `msrun startup
            <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 4 devices.

        >>> import numpy as np
        >>> import os
        >>> import math
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, Parameter, train
        >>> from mindspore.communication import init, get_rank
        >>> from mindspore.common.initializer import initializer, HeUniform
        >>> from mindspore.train import Model, TrainFaultTolerance
        >>> from mindspore import dataset as ds
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
        >>> init()
        >>> ms.set_seed(1)
        >>> ms.set_auto_parallel_context(strategy_ckpt_config={"save_file":
        ...                             "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
        >>> class MatMulCell(nn.Cell):
        ...     def __init__(self, param=None, shape=None):
        ...         super().__init__()
        ...         if shape is None:
        ...             shape = [28 * 28, 512]
        ...         weight_init = HeUniform(math.sqrt(5))
        ...         self.param = Parameter(initializer(weight_init, shape), name="param")
        ...         if param is not None:
        ...             self.param = param
        ...         self.print = ops.Print()
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         out = self.matmul(x, self.param)
        ...         self.print("out is:", out)
        ...         return out
        >>>
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.flatten = nn.Flatten()
        ...         self.layer1 = MatMulCell()
        ...         self.relu1 = nn.ReLU()
        ...         self.layer2 = nn.Dense(512, 512)
        ...         self.relu2 = nn.ReLU()
        ...         self.layer3 = nn.Dense(512, 10)
        ...
        ...     def construct(self, x):
        ...         x = self.flatten(x)
        ...         x = self.layer1(x)
        ...         x = self.relu1(x)
        ...         x = self.layer2(x)
        ...         x = self.relu2(x)
        ...         logits = self.layer3(x)
        ...         return logits
        >>>
        >>> net = Network()
        >>> net.layer1.pipeline_stage = 0
        >>> net.relu1.pipeline_stage = 0
        >>> net.layer2.pipeline_stage = 0
        >>> net.relu2.pipeline_stage = 1
        >>> net.layer3.pipeline_stage = 1
        >>>
        >>> def create_dataset(batch_size):
        ...     dataset_path = os.getenv("DATA_PATH")
        ...     dataset = ds.MnistDataset(dataset_path)
        ...     image_transforms = [
        ...         ds.vision.Rescale(1.0 / 255.0, 0),
        ...         ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ...         ds.vision.HWC2CHW()
        ...     ]
        ...     label_transform = ds.transforms.TypeCast(ms.int32)
        ...     dataset = dataset.map(image_transforms, 'image')
        ...     dataset = dataset.map(label_transform, 'label')
        ...     dataset = dataset.batch(batch_size)
        ...     return dataset
        >>>
        >>> dataset = create_dataset(32)
        >>>
        >>> optimizer = nn.SGD(net.trainable_params(), 1e-2)
        >>> optimizer_wrapper = nn.OptTFTWrapper(optimizer)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>>
        >>> net_with_loss = nn.Pipeline(nn.WithLossCell(net, loss_fn), 4)
        >>> net_with_loss.set_train()
        >>> model = Model(net_with_loss, optimizer=optimizer_wrapper)
        >>> tft_cb = TrainFaultTolerance()
        >>> loss_cb = train.LossMonitor(1)
        >>> model.train(1, dataset, callbacks=[tft_cb, loss_cb])
    """

    def __init__(self, ckpt_save_path=None, **kwargs):
        super().__init__()
        self.envs = None
        if self._only_enable_ato():
            ato_path, ato_threshold = self._get_ato_env_args()
            self.ato = AdaptTrainOptimizer(ato_path, ato_threshold)
            return
        if self._only_enable_tsp():
            self.tft = _tft_handler.get_tft()
            self._check_init()
            self.tft.tft_register_stream_sync_handler(runtime.synchronize, self)
            return
        self.save_cb = kwargs.get("ckpt_save_fn", None)
        self.ckpt_save_path = ckpt_save_path
        if self.save_cb is None and self.ckpt_save_path is None:
            raise ValueError("TrainFaultTolerance construct need to set ckpt_save_fn or ckpt_save_path!")
        self.cb_params = None
        self.initial_step = kwargs.get("initial_step", 0)
        self.device_id = context.get_context("device_id")
        self.cur_step_num = 0
        self.cur_epoch_num = 0
        # For TREError(Training Result Error) scene, parameter `ckpt_load_fn` must be provided to load checkpoint
        # from file for resuming training, the `ckpt_load_fn` is a function, prototype of which is:
        # `def load_checkpoint() -> tuple(dict, bool)`, the return value is a tuple containing 2 values,
        # i.e. (param_dict, remove_redundancy)
        self.ckpt_load_func = kwargs.get("ckpt_load_fn", None)
        if self._only_enable_tre() or self._only_enable_ckpt_d2h_async():
            return
        self.tft = _tft_handler.get_tft()
        self._check_init()
        if self._only_enable_tre_and_tsp():
            self.tft.tft_register_stream_sync_handler(runtime.synchronize, self)
            return
        self.global_step = None
        self.learning_rate = None
        self.has_init_replica = False
        self.is_uce_rank = False

        self.assign = mindspore.ops.Assign()
        self.g_one = Tensor([1], dtype=mstype.int32)
        _tft_sem_enable()
        self._tft_register()

    def _only_enable_ato(self):
        """Check whether ATO mode is enabled without other TFT modes."""
        env_enable = os.getenv("MS_ENABLE_TFT", "")
        ato_env = "ATO:1" in env_enable
        non_ato_flags = ["TTP:1", "UCE:1", "ARF:1", "TRE:1", "TSP:1"]
        if any(flag in env_enable for flag in non_ato_flags):
            if ato_env:
                raise ValueError("ATO checkpoint optimization is enabled but other non-ATO flags are also set, "
                               "which may cause unexpected behavior. Please check environment variable MS_ENABLE_TFT.")
            return False
        return ato_env

    def _get_ato_env_args(self):
        """Get ATO enable flag and initialization arguments from `MS_ENABLE_TFT`."""
        env_enable = os.getenv("MS_ENABLE_TFT", "")
        env_content = env_enable.strip()
        if env_content.startswith("{") and env_content.endswith("}"):
            env_content = env_content[1:-1]
        env_config = {}
        for item in env_content.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            key, value = item.split(":", 1)
            env_config[key.strip()] = value.strip()

        ato_threshold_str = env_config.get("ATO_THRESHOLD")
        if ato_threshold_str is None:
            ato_threshold = 10
        else:
            if ato_threshold_str == "":
                raise ValueError("ATO_THRESHOLD in MS_ENABLE_TFT cannot be empty")
            try:
                ato_threshold = int(ato_threshold_str)
            except ValueError as e:
                raise ValueError(f"ATO_THRESHOLD must be an integer, got '{ato_threshold_str}'") from e
            if ato_threshold < 10:
                raise ValueError("For ATO, 'ATO_THRESHOLD' in 'MS_ENABLE_TFT' must be "
                                 "an integer greater than or equal to 10.")
        ato_path = env_config.get("ATO_PATH")
        if ato_path is None:
            ato_path = "./ato_info.json"
        else:
            if ato_path == "":
                raise ValueError("ATO_PATH in MS_ENABLE_TFT cannot be empty")
            if not ato_path.endswith(".json"):
                raise ValueError(f"ATO_PATH must be a JSON file path, got '{ato_path}'")
        return ato_path, ato_threshold

    def _only_enable_tre(self):
        """Check if only configured MS_ENABLE_TFT='{TRE:1}'"""
        env_enable = os.getenv("MS_ENABLE_TFT", "")
        non_tre_flags = ["TTP:1", "UCE:1", "ARF:1"]
        if any(flag in env_enable for flag in non_tre_flags):
            return False
        return "TRE:1" in env_enable or "TRE:2" in env_enable

    @staticmethod
    def _only_enable_ckpt_d2h_async():
        """Check whether only set MS_ENABLE_CKPT_D2H_ASYNC=1 without setting MS_ENABLE_TFT"""
        if os.getenv("MS_ENABLE_TFT", "") != "":
            return False
        return os.getenv("MS_ENABLE_CKPT_D2H_ASYNC") == "1"

    @staticmethod
    def _enable_snapshot():
        """Check whether parameter snapshot is enabled"""
        enable_step_tre = "TRE:2" in os.getenv("MS_ENABLE_TFT", "")
        enable_ckpt_d2h_async = os.getenv("MS_ENABLE_CKPT_D2H_ASYNC") == "1"
        return enable_step_tre or enable_ckpt_d2h_async

    def _only_enable_tsp(self):
        """Check if only configured MS_ENABLE_TFT='{TSP:1}'"""
        env_enable = os.getenv("MS_ENABLE_TFT", "")
        non_tsp_flags = ["TTP:1", "UCE:1", "ARF:1", "TRE:1"]
        if any(flag in env_enable for flag in non_tsp_flags):
            return False
        return "TSP:1" in env_enable

    def _only_enable_tre_and_tsp(self):
        """Check if only configured MS_ENABLE_TFT='{TRE:1, TSP:1}'"""
        env_enable = os.getenv("MS_ENABLE_TFT", "")
        other_flags = ["TTP:1", "UCE:1", "ARF:1"]
        if any(flag in env_enable for flag in other_flags):
            return False
        return "TRE:1" in env_enable and "TSP:1" in env_enable

    def _check_init(self):
        """Check if the mindio-ttp had initialized"""
        if self.tft is None:
            tft_env = os.getenv("MS_ENABLE_TFT", "")
            if "ARF:1" in tft_env:
                raise ValueError("Must init by _tft_handler.init(config=params) if use ARF.")
            logger.warning("TFT handle not init, try to init")
            _tft_handler.init(config=None)
            self.tft = _tft_handler.get_tft()
            logger.warning("TFT handle init ok.")
        device_target = context.get_context("device_target")
        if device_target != "Ascend":
            raise ValueError(f"MindIO adapter only support on Ascend device but got device {device_target}!")

    def _is_params_consistent(self):
        for key, param in self.cb_params.train_network.parameters_and_names():
            if "tft_g_one_flag" in key:
                tft_g_one_flag = direct_copy_to_host(param)
                return int(tft_g_one_flag) == 1
        return False

    def _set_tft_optimizer_replica(self, run_context):
        """Set Mindio TFT optimizer replica info, used internally."""
        cur_rank = get_rank()
        cb_params = run_context.original_args()
        train_network = cb_params.train_network
        # in data_parallel mode, every ranks has same train parameters
        if context.get_auto_parallel_context("parallel_mode") == "data_parallel":
            group_size = get_group_size()
            dp = tuple(range(group_size))
        else:
            param_layout_dict = train_network.parameter_layout_dict
            dp = _get_cur_rank_dp(param_layout_dict) if param_layout_dict else _get_cur_rank_dp(train_network)
        logger.warning(f"Set TFT replica with dp: {dp}.")
        replica_info = [
            {
                "type": 1,
                "rank_list": list(dp),
                "replica_cnt": len(dp),
                "replica_shift": 0
            }
        ]
        self.tft.tft_set_optimizer_replica(cur_rank, replica_info)

    @classmethod
    def get_optimizer_wrapper(cls, origin_opt_cls):
        """
        Optimizer wrapper func when using tft.

        Args:
            origin_opt_cls (Class): origin optimizer class.
        """

        class TFTOptSubCls(origin_opt_cls):
            """
            Optimizer wrapper class when using tft.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.report = TensorReport()
                self.report_end = TensorReport()
                self.report_end.add_prim_attr("optimizer_end", True)
                self.depend = ops.Depend()
                self.allreduce_sum = ops.AllReduce()
                self.allreduce_sum.add_prim_attr("tft_report_before", True)
                self.tft_g_one_flag = Parameter(Tensor([1], dtype=mstype.int32))

            def construct(self, gradients, **kwargs):
                tft_g_one_flag = self.depend(self.tft_g_one_flag, gradients)
                self.tft_g_one_flag = self.allreduce_sum(tft_g_one_flag)
                grads = self.depend(gradients, self.report("tft_report", self.tft_g_one_flag))
                opt_ret = super().construct(grads, **kwargs)
                self.report_end("tft_report", self.tft_g_one_flag)
                return opt_ret

        class TFTOptSnapShotCls(origin_opt_cls):
            """
            Optimizer wrapper class when using tft.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.report = TensorReport()
                self.report.add_prim_attr("side_effect_mem", True).add_prim_attr("snapshot", True)
                self.dummy_input = Tensor([1], dtype=mstype.int32)

            def construct(self, gradients, **kwargs):
                """Add fake op TensorReport to insert wait event for copying parameters"""
                self.report("tft_report", self.dummy_input)
                opt_ret = super().construct(gradients, **kwargs)
                return opt_ret

        env_tft = os.getenv('MS_ENABLE_TFT', '')
        features = ['TTP:1', 'UCE:1', 'ARF:1', 'HCCE:1']
        need_redundancy = any([env_tft.find(feat) >= 0 for feat in features])  # pylint: disable=R1729
        return TFTOptSubCls if need_redundancy else TFTOptSnapShotCls

    def _tft_register(self):
        """Register callback functions."""
        self.tft.tft_register_save_ckpt_handler(_save_checkpoint_on_failure, self)
        self.tft.tft_register_rename_handler(_rename_save_result, self)
        self.tft.tft_register_exit_handler(_tft_exit_cb, self)
        self.tft.tft_register_stop_handler(_tft_stop_callback, self)
        self.tft.tft_register_clean_handler(_tft_clean_callback, self)
        self.tft.tft_register_repair_handler(_tft_repair_callback, self)
        self.tft.tft_register_rebuild_group_handler(_tft_rebuild_groups, self)
        if "TSP:1" in os.getenv("MS_ENABLE_TFT", ""):
            self.tft.tft_register_stream_sync_handler(runtime.synchronize, self)

    def _reset_acc_grads(self):
        accu_grad_params = map(lambda e: e[1],
                               filter(lambda e: e[1].name.startswith('accu_grads'),
                                      self.cb_params.train_network.parameters_and_names()))
        accu_grad_list = list(accu_grad_params)
        if reset_params(accu_grad_list) != 0:
            raise ValueError("Call reset_params failed.")

    def on_train_step_begin(self, run_context):
        """
        Clear saving snapshot state at each step begin.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for details.
        """
        if self._only_enable_ato():
            self.ato.step_start_time = time.time()
            return
        if self._enable_snapshot():
            _clear_snapshot_saving_flag()

    def on_train_step_end(self, run_context):
        """
        Report status to MindIO TFT after every step finished.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for details.
        """
        if self._only_enable_ato():
            cb_params = run_context.original_args()
            if self.ato.step_start_time is not None:
                step_time = time.time() - self.ato.step_start_time
                if cb_params.dataset_sink_mode and cb_params.batch_num:
                    step_time = step_time / cb_params.batch_num
                self.ato.step_times.append(step_time)
            return
        if self._only_enable_tre() or self._only_enable_ckpt_d2h_async():
            return

        cb_params = run_context.original_args()
        logger.info("START Set optimizer finish step status to TFT. step: {}".format(cb_params.cur_step_num))
        self.cur_step_num = cb_params.cur_step_num
        self.cur_epoch_num = cb_params.cur_epoch_num
        if self._only_enable_tsp() or self._only_enable_tre_and_tsp():
            logger.info("Go into tft_pause_train.")
            self.tft.tft_pause_train(self.cur_step_num)
            return
        if self.has_init_replica is False:
            self.has_init_replica = True
            # stable after training
            self.envs = os.getenv("MS_ENABLE_TFT", "")
            self._set_tft_optimizer_replica(run_context)
        self._end_update_report(cb_params)
        self._reset_arf_on_step_end(run_context)

    def _get_optim(self, cb_params):
        """Get optimizer from cb_params"""
        if cb_params.optimizer is not None:
            return cb_params.optimizer
        if hasattr(cb_params.network, 'optimizer') and cb_params.network.optimizer is not None:
            return cb_params.network.optimizer
        raise ValueError("TFT feature need optimizer or network's optimizer!")

    def _end_update_report(self, cb_params):
        """check overflow: no need end update and pause train if training overflow"""
        optim = self._get_optim(cb_params)
        self.global_step = optim.global_step.clone()

        def _reset_status():
            """reset tft_g_one_flag"""
            self.assign(optim.tft_g_one_flag, self.g_one)

        if self.envs is None:
            self.envs = os.getenv("MS_ENABLE_TFT", "")
        if int(direct_copy_to_host(optim.tft_g_one_flag)) != 1:
            self.tft.tft_end_updating_os(self.cur_step_num + self.initial_step)
            logger.info("End updating step to tft.")
            # pause train
            if any(opt in self.envs for opt in ["TSP:1", "ARF:1"]):
                _reset_status()
                logger.info("Go into tft_pause_train.")
                self.tft.tft_pause_train(self.cur_step_num + self.initial_step)
                logger.info("End tft_pause_train.")
        _reset_status()

    def _reset_arf_on_step_end(self, run_context):
        """reset arf flag on train step end"""
        cb_params = run_context.original_args()
        if cb_params.is_arf:
            cb_params.is_arf = False
            set_is_arf(False)

    def on_train_begin(self, run_context):
        """
        Register train params to MindIO TFT on train beginning.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                :class:`mindspore.train.RunContext` for details.
        """
        cb_params = run_context.original_args()
        self.cb_params = cb_params
        _reset_opt_event_info()
        if self._only_enable_ato():
            return
        if self._enable_snapshot():
            param_dict = {}
            for param in cb_params.train_network.trainable_params():
                param_dict[param.name] = param
            _reg_snapshot_params(param_dict)
        if self._only_enable_tsp():
            return
        if self._only_enable_tre() or self._only_enable_ckpt_d2h_async():
            return
        sink_size = cb_params.get("sink_size", 0)
        if sink_size > 1:
            raise ValueError("TFT feature doesn't support sink_size > 1.")
        logger.info("Set args to TFT.")
        self.tft.tft_set_step_args(cb_params)
        cb_params.is_arf = check_is_arf()

    def end(self, run_context):
        """
        Unregister MindIO TFT on train end.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for details.
        """
        enable_flags = (
        self._only_enable_tre(),
        self._only_enable_tsp(),
        self._only_enable_tre_and_tsp(),
        self._only_enable_ckpt_d2h_async(),
        self._only_enable_ato()
        )

        if any(enable_flags):
            return
        _tft_handler.unregister_tft()
