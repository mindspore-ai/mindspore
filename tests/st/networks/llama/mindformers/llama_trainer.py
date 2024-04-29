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
"""Base Trainer."""
import os
import time
from typing import Optional, Union

import mindspore as ms
from mindspore.nn import Cell, \
    PipelineCell, MicroBatchInterleaved
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.train.model import Model

try:
    # new interface in ms2.1.1
    from mindspore.nn.wrap.cell_wrapper import GradAccumulationCell

    GRAD_ACCUMULATION_VALID = True
except ImportError:
    GRAD_ACCUMULATION_VALID = False

from mindformers import CosineWithWarmUpLR, FP32StateAdamWeightDecay, MFTrainOneStepCell, MFPipelineWithLossScaleCell
from mindformers.core.optim.llama_optim import get_optimizer_grouped_parameters
from mindformers.parallel_config import build_parallel_config
from mindformers.llama_utils import check_runner_config
from mindformers.llama_utils import get_real_group_size
from mindformers import LlamaCallback

import logging
logger = logging.getLogger()

wrapper_map = {"MFTrainOneStepCell": MFTrainOneStepCell,
               "MFPipelineWithLossScaleCell": MFPipelineWithLossScaleCell}


class LlamaTrainer:
    r"""Base Task Trainer.
    Args:
        task (str): The task name supported.
        model_name (str): The model name of Task-Trainer. Default: None
    Raises:
        NotImplementedError: If train method or evaluate method or predict method not implemented.
    """

    def __init__(self, task: str = None, model_name: str = None):

        if model_name is None:
            model_name = "model name unspecified."
        if task is None:
            task = "task name unspecified."
        logger.info("Now Running Task is: %s, Model is: %s", task, model_name)

        self.model_name = model_name
        self.task = task
        self.config = None
        self.default_task_config = None

        self.train_dataset = None
        self.eval_dataset = None
        self.network = None
        self.optimizer = None
        self.image_processor = None
        self.audio_processor = None
        self.tokenizer = None
        self.callbacks = None
        self.eval_callbacks = None
        self.model_wrapper = None
        self.compute_metrics = None
        self.kwargs = None
        self.pipeline_task = None
        self.callbacks = []
        self.compile_time = []

    def set_config(self,
                   config: Optional[Union[dict, str]] = None,
                   is_full_config: bool = False):
        """Set the task config for task trainer."""
        self.config = config
        build_parallel_config(self.config)
        self._check_grad_accumulation_steps()
        self._check_global_batch_size_for_auto_parallel()

        return self.config

    def _check_global_batch_size_for_auto_parallel(self):
        """Check global batch size in auto parallel mode."""
        batch_size = self.config.runner_config.batch_size
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        dp = self.config.parallel_config.data_parallel
        micro_batch_num = self.config.parallel_config.micro_batch_num
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        full_batch = ms.get_auto_parallel_context("full_batch")
        pp = self.get_pipeline_stages()

        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            if full_batch:
                if pp > 1:
                    self.global_batch_size = batch_size * dp * micro_batch_num * micro_batch_interleave_num
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is True, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "global batch size will be changed: "
                                "global_batch_size = "
                                "batch_size * data_parallel * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s * %s).",
                                pp, self.global_batch_size, batch_size, dp, micro_batch_num,
                                micro_batch_interleave_num)
                    self.config.runner_config.batch_size = self.global_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    self.global_batch_size = batch_size * dp * micro_batch_interleave_num * gradient_accumulation_steps
                    logger.info("The current parallel mode is %s, full batch is True,"
                                "so global batch size will be changed: "
                                "global_batch_size = batch_size * data_parallel * micro_batch_interleave_num "
                                "* gradient_accumulation_steps = %s = %s * %s * %s * %s",
                                parallel_mode, self.global_batch_size, batch_size, dp, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    self.config.runner_config.batch_size = self.global_batch_size
            else:
                if pp > 1:
                    per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num
                    self.global_batch_size = per_batch_size * get_real_group_size()
                    logger.info("Pipeline parallel was opened: pipeline_stages = %s, full batch is False, "
                                "gradient_accumulation_steps will not take effect in pipeline parallel, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_num * micro_batch_interleave_num "
                                "= %s = %s * %s * %s).",
                                pp, per_batch_size, batch_size, micro_batch_num,
                                micro_batch_interleave_num)
                    logger.info("global_batch_size = per_batch_size * device_num = %s * %s = %s",
                                per_batch_size, get_real_group_size(), self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
                    self._reset_wrapper_for_pipeline_parallel()
                else:
                    per_batch_size = batch_size * micro_batch_interleave_num * gradient_accumulation_steps
                    self.global_batch_size = per_batch_size * get_real_group_size()
                    logger.info("The current parallel mode is %s, full batch is False, "
                                "batch size per card will be changed: "
                                "per_batch_size = batch_size * micro_batch_interleave_num * "
                                "gradient_accumulation_steps = %s = %s * %s * %s).",
                                parallel_mode, per_batch_size, batch_size, micro_batch_interleave_num,
                                gradient_accumulation_steps)
                    logger.info("global_batch_size = per_batch_size * device_num = %s * %s = %s",
                                per_batch_size, get_real_group_size(), self.global_batch_size)
                    self.config.runner_config.batch_size = per_batch_size
            if gradient_accumulation_steps > 1:
                self._reset_wrapper_for_grad_accu()
        else:
            logger.info("The current parallel mode is %s, batch size per card will not be changed: "
                        "batch_size_per_card = %s",
                        parallel_mode, batch_size)
            self.global_batch_size = batch_size * get_real_group_size() * gradient_accumulation_steps
            logger.info(
                "global_batch_size = batch_size_per_card * device_num * gradient_accumulation_steps "
                "= %s = %s * %s * %s",
                self.global_batch_size, batch_size, get_real_group_size(), gradient_accumulation_steps)
            self.config.runner_config.batch_size = batch_size * gradient_accumulation_steps
            self.config.parallel_config.data_parallel = 1
            self.config.parallel_config.model_parallel = 1
            self.config.parallel_config.pipeline_stage = 1
            self.config.parallel_config.micro_batch_num = 1
            logger.info("parallel_config will be change to default config: %s.",
                        self.config.parallel_config)

    def _check_grad_accumulation_steps(self):
        """check the gradient accumulation steps."""
        if self.config.runner_config.gradient_accumulation_steps is None:
            self.config.runner_config.gradient_accumulation_steps = 1
        if not isinstance(self.config.runner_config.gradient_accumulation_steps, int) or \
                isinstance(self.config.runner_config.gradient_accumulation_steps, bool):
            raise ValueError("gradient_accumulation should be integer but got "
                             f"{type(self.config.runner_config.gradient_accumulation_steps)}")
        if not self.config.runner_config.gradient_accumulation_steps >= 1:
            raise ValueError("gradient_accumulation should be greater or equal than 1, "
                             f"but got {self.config.runner_config.gradient_accumulation_steps}")
        if not GRAD_ACCUMULATION_VALID and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps only surpport mindspore version later than 2.1.1, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"] and pp > 1 \
                and self.config.runner_config.gradient_accumulation_steps > 1:
            logger.warning("gradient_accumulation_steps will not take effect when using pipeline parallel, "
                           "reset the gradient_accumulation_steps from %s to 1.",
                           self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1
        # grad accumulation not supported in data parallel/standalone mode for now
        if self.config.runner_config.gradient_accumulation_steps > 1 and \
                parallel_mode not in ["semi_auto_parallel"]:
            logger.warning("gradient_accumulation_steps currently need to be used in semi_auto_parallel mode, "
                           "but got %s mode, please check your runner config and parallel config. "
                           "Reset the gradient_accumulation_steps from %s to 1. ",
                           parallel_mode, self.config.runner_config.gradient_accumulation_steps)
            self.config.runner_config.gradient_accumulation_steps = 1

    def _check_training_network_no_use_past(self, network):
        if network is not None and hasattr(network.config, "use_past") and network.config.use_past:
            raise ValueError("In training process, network should be configured to use_past=False, "
                             f"but got use_past={network.config.use_past}")

    def _reset_wrapper_for_pipeline_parallel(self):
        """Reset wrapper when pipeline parallel."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
            self.config.runner_wrapper.micro_batch_num = self.config.parallel_config.micro_batch_num
            logger.warning(
                "When using the pipeline parallel mode, "
                "the MFPipelineWithLossScaleCell class is used by default.")
        else:
            logger.info(
                "When using the pipeline parallel mode, "
                "because the wrapper class is not specified, "
                "MindSpore's built-in PipelineCell is used by default")
        logger.info("PipelineWrapper under evaluate or predict mode will not take effect.")

    def _reset_wrapper_for_grad_accu(self):
        """Reset wrapper when using grad accumulation."""
        if self.config.runner_wrapper is not None:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell" \
                if self.config.runner_wrapper.type != "MFPipelineWithLossScaleCell" else self.config.runner_wrapper.type
        else:
            self.config.runner_wrapper.type = "MFPipelineWithLossScaleCell"
        logger.warning(
            "When using the gradient_accumulation_steps in semi/auto parallel mode, "
            "the MFPipelineWithLossScaleCell class is used by default.")

    def wrap_network_with_tool_cells(self, network):
        """For training process, warp the network with some tool cells."""
        micro_batch_interleave_num = self.config.micro_batch_interleave_num
        gradient_accumulation_steps = self.config.runner_config.gradient_accumulation_steps
        parallel_mode = ms.context.get_auto_parallel_context("parallel_mode")
        pp = self.get_pipeline_stages()
        if micro_batch_interleave_num > 1:
            logger.info("micro_batch_interleave_num > 1, the double copy parallel feature is turned on.")
            network = MicroBatchInterleaved(network, micro_batch_interleave_num)
        if gradient_accumulation_steps > 1 and not pp > 1:
            logger.info("gradient_accumulation_steps > 1, GradAccumulationCell is wrapped on network. "
                        "It is suggested to use `Lazy Inline` feature to save compiling time.")
            network = GradAccumulationCell(network, gradient_accumulation_steps)
        if pp > 1:
            micro_batch_num = self.config.parallel_config.micro_batch_num
            network = PipelineCell(network, micro_size=micro_batch_num)
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            network = _VirtualDatasetCell(network)
        return network

    def create_optimizer_scheduler(self, network, layer_scale=False):
        """Create the optimizer for training."""
        # learning rate scale for multi-nodes training
        lr_schedule = self.create_lr_scheduler()
        weight_decay = self.config.optimizer.weight_decay if self.config.optimizer.weight_decay else 0.
        layer_decay = self.config.layer_decay if self.config.layer_decay else 1.0
        group_params = get_optimizer_grouped_parameters(network,
                                                        weight_decay,
                                                        lr_schedule,
                                                        layer_scale=layer_scale,
                                                        layer_decay=layer_decay)
        self.config.optimizer.learning_rate = lr_schedule
        self.config.optimizer.pop("type")
        optimizer = FP32StateAdamWeightDecay(group_params, **self.config.optimizer)
        return optimizer

    def create_lr_scheduler(self):
        # learning rate scale for multi-nodes training
        learning_scale = self.config.lr_scale
        scale_factor = self.config.lr_scale_factor

        train_data_size = self.train_dataset.get_dataset_size()

        if self.config.lr_schedule:
            warmup_epochs = self.config.lr_schedule.pop("warmup_epochs", None)
            warmup_ratio = self.config.lr_schedule.pop("warmup_ratio", None)

            if not self.config.runner_config.sink_mode:
                total_steps = int(self.config.runner_config.epochs * train_data_size)
            else:
                total_steps = int(self.config.runner_config.epochs * self.config.runner_config.sink_size)
            if warmup_epochs is not None and warmup_ratio is not None:
                warmup_epochs = None

            if warmup_epochs is not None:
                self.config.lr_schedule.warmup_steps = int(warmup_epochs * train_data_size)

            if warmup_ratio is not None:
                self.config.lr_schedule.warmup_steps = int(total_steps * warmup_ratio)
            self.config.lr_schedule.total_steps = total_steps \
                if self.config.lr_schedule.total_steps is None or self.config.lr_schedule.total_steps == -1 \
                else int(self.config.lr_schedule.total_steps)
            if learning_scale and scale_factor is not None:
                device_num = get_real_group_size()
                per_device_batch_size = self.train_dataset.batch_size
                self.config.lr_schedule.learning_rate = (self.config.lr_schedule.learning_rate *
                                                         device_num * per_device_batch_size) / scale_factor

        lr_schedule = CosineWithWarmUpLR(**self.config.lr_schedule)
        return lr_schedule

    def set_train_dataset(self, dataset):
        """Set the attribute of train dataset."""
        if dataset is None:
            raise ValueError("Train dataset is None")
        self.train_dataset = dataset

    def set_network(self, network, is_train: bool = True):
        """Set the attribute of network."""
        if network is None:
            raise ValueError("network is None")
        if isinstance(network, (Cell)):
            network.set_train(is_train)
        self.network = network

    def get_train_data_size(self):
        """Get train dataset size."""
        if self.train_dataset is None:
            raise NotImplementedError("train dataset is None")
        return self.train_dataset.get_dataset_size()

    def get_pipeline_stages(self):
        """Get pipeline stages for task trainer."""
        pipeline_stages = ms.get_auto_parallel_context("pipeline_stages")
        return pipeline_stages

    def learning_rate_scale(self, base_learning_rate: float = 0., scale_factor: Optional[Union[float, int]] = 256.):
        """Scale learning rate for training."""
        if not isinstance(base_learning_rate, float):
            raise ValueError(f"learning rate must be float type, but get {type(base_learning_rate)}")
        if not isinstance(scale_factor, (float, int)):
            raise ValueError(f"scale_factor must be float or int type, but get {type(scale_factor)}")

        device_num = get_real_group_size()
        per_device_batch_size = self.config.train_dataset.batch_size
        learning_rate = (base_learning_rate * device_num * per_device_batch_size) / scale_factor
        return learning_rate

    def create_model_wrapper(self, network, optimizer):
        wrapper_type = self.config.runner_wrapper.pop("type")
        wrapper = wrapper_map[wrapper_type](network=network, optimizer=optimizer,
                                            parallel_config=self.config.parallel_config, **self.config.runner_wrapper)
        return wrapper

    def train(self,
              config=None,
              network=None,
              dataset=None,
              optimizer=None,
              wrapper=None,
              callbacks=None,
              **kwargs):
        """Train or Fine-tune for BaseTainer in MindFormers."""
        self.kwargs = kwargs

        config.runner_config.initial_epoch = 0
        config.runner_config.initial_step = 0

        self.set_train_dataset(dataset)
        check_runner_config(config, dataset)

        network = self.network

        if network is not None:
            eval_network = network
            # warp network for training
            network = self.wrap_network_with_tool_cells(eval_network)
            self.set_network(network, is_train=True)

        optimizer = self.create_optimizer_scheduler(network, layer_scale=config.layer_scale)

        # build model wrapper
        # logger.info("...***..Build Running Wrapper From Config For Train..........")
        wrapper = self.create_model_wrapper(network, optimizer)

        default_args = {
            "learning_rate": optimizer.learning_rate if optimizer else wrapper.optimizer.learning_rate,
            "origin_epochs": config.runner_config.origin_epochs,
            "dataset_size": config.data_size,
            "micro_batch_interleave_num": config.micro_batch_interleave_num,
            "micro_batch_num": config.parallel_config.micro_batch_num,
            "initial_epoch": config.runner_config.initial_epoch,
            "initial_step": config.runner_config.initial_step,
            "global_batch_size": self.global_batch_size,
            "gradient_accumulation_steps": self.config.runner_config.gradient_accumulation_steps
        }
        callbacks = LlamaCallback(**default_args)
        self.callbacks.append(callbacks)

        model = Model(wrapper, metrics=None, eval_network=None)

        is_lazyinline = os.getenv("ENABLE_CELL_REUSE", "0")
        pipline_parallel = ms.get_auto_parallel_context("pipeline_stages") > 1
        if is_lazyinline and pipline_parallel:
            start_time = time.perf_counter()
            model.build(train_dataset=dataset, epoch=config.runner_config.epochs,
                        sink_size=config.runner_config.sink_size)
            end_time = time.perf_counter()
            self.compile_time.append(end_time - start_time)

            print(f"Compile time: {self.compile_time[0]}")

        model.train(config.runner_config.epochs, dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=config.runner_config.sink_mode,
                    sink_size=config.runner_config.sink_size,
                    initial_epoch=config.runner_config.initial_epoch)
