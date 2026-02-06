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
# ============================================================================
"""Model training base utilities for distributed training tests in MindSpore.

This module provides a base class for creating, training, and evaluating MindSpore models
with checkpoint management capabilities. It simplifies the process of setting up training
pipelines with automatic checkpoint saving/loading, mixed precision training (AMP), and
distributed inference with multi-rank checkpoint support.

Key Features:
- Model creation with customizable loss functions and optimizers
- Automatic checkpoint saving during training with configurable intervals
- Latest checkpoint loading after training completion
- Asynchronous checkpoint saving for non-blocking training
- Distributed checkpoint loading for multi-rank inference
- Mixed precision training support (AMP levels: O0, O1, O2, O3)
- Dataset sink mode support for performance optimization
- Distributed training with custom communication strategies
- Automatic checkpoint cleanup to prevent disk space issues

Classes:
- ModelTrainBase: Main utility class for model training and checkpoint management

Usage:
    This module is primarily used in distributed training tests to standardize the model
    training workflow and checkpoint handling across different test scenarios.

Example:
    # Create trainer instance
    trainer = ModelTrainBase()
    
    # Create and train model with checkpoint saving
    model = trainer.create_train_model(network, loss=loss_fn, opt=optimizer)
    ckpt_dict = trainer.load_newest_ckpt_from_model_train(
        model, epoch=10, dataset=train_dataset, ckpt_path="./checkpoints/")
    
    # Parallel inference with distributed checkpoints
    output = trainer.create_parallel_model_and_predict(
        network, ckpt_file_list, predict_data)
"""
from mindspore import log as logger
from mindspore import context
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint
from mindspore import load_distributed_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint_async
from tests.st.auto_parallel.utils.utils import clean_all_ckpt_files, find_newest_ckpt_file, find_newest_ckpt_file_by_name


class ModelTrainBase():
    def __init__(self):
        self.save_ckpt = False
        self.ckpt_path = ''

    def __del__(self):
        if self.save_ckpt:
            clean_all_ckpt_files(self.ckpt_path)

    # create a model
    def create_train_model(self, network, amp_level="O0", metrics=None, loss_scale_manager=None,
                           loss="default", opt=None):
        logger.info(f"MindSporeTest::create a model with amp_level={amp_level}")
        if loss == "default":
            loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        opt_fn = opt
        if opt_fn is None:
            opt_fn = Momentum(learning_rate=0.01, momentum=0.9, params=network.get_parameters())
        model = Model(network=network, loss_fn=loss, optimizer=opt_fn, amp_level=amp_level,
                      metrics=metrics, loss_scale_manager=loss_scale_manager)

        return model

    # save checkpoint when model train, load the newest checkpoint
    def load_newest_ckpt_from_model_train(self, model, epoch, dataset, dataset_sink_mode=True, user_callback=None,
                                          ckpt_path="./", ckpt_prefix="ckpt_ms", async_save=False,
                                          save_checkpoint_steps=1, sink_size=-1,
                                          integrated_save=True, format_="ckpt",
                                          load_format="default"):
        if not user_callback:
            user_callback = []
        logger.info("MindSporeTest::configure Config to save Checkpoint")
        ckpt_config = CheckpointConfig(keep_checkpoint_max=5, integrated_save=integrated_save,
                                       save_checkpoint_steps=save_checkpoint_steps,
                                       async_save=async_save, format=format_)
        ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)

        logger.info(f"MindSporeTest::clean all Checkpoint file under {ckpt_path}")
        clean_all_ckpt_files(ckpt_path)

        logger.info(f"MindSporeTest::Model train and save checkpoint under {ckpt_path}")
        callbacks = [ckpt_callback]
        callbacks.extend(user_callback)
        model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                    callbacks=callbacks, sink_size=sink_size)

        logger.info("MindSporeTest::load the newest checkpoint file and return")
        if load_format == "default":
            newest_ckpt_file = find_newest_ckpt_file(ckpt_path, format_=format_)
        else:
            newest_ckpt_file = find_newest_ckpt_file_by_name(ckpt_path, format_=format_)
        self.save_ckpt = True
        self.ckpt_path = ckpt_path
        return load_checkpoint(newest_ckpt_file, format=format_)

    def load_newest_ckpt_async_from_model_train(self, model, epoch, dataset, dataset_sink_mode=True,
                                                ckpt_path="./", ckpt_prefix="ckpt_ms",
                                                async_save=False,
                                                save_checkpoint_steps=1, sink_size=-1,
                                                integrated_save=True, build_dataset=None,
                                                load_format="default"):
        logger.info("MindSporeTest::configure Config to save Checkpoint")
        ckpt_config = CheckpointConfig(keep_checkpoint_max=5, integrated_save=integrated_save,
                                       save_checkpoint_steps=save_checkpoint_steps,
                                       async_save=async_save)
        ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)

        logger.info(f"MindSporeTest::clean all Checkpoint file under {ckpt_path}")
        clean_all_ckpt_files(ckpt_path)

        logger.info(f"MindSporeTest::Model train and save checkpoint under {ckpt_path}")
        model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                    callbacks=[ckpt_callback], sink_size=sink_size)

        logger.info("MindSporeTest::load the newest checkpoint file and return")
        if load_format == "default":
            newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
        else:
            newest_ckpt_file = find_newest_ckpt_file_by_name(ckpt_path)
        self.save_ckpt = True
        self.ckpt_path = ckpt_path
        frature = load_checkpoint_async(newest_ckpt_file)
        model.build(train_dataset=build_dataset, epoch=5)
        return frature.result()

    # create a model and train 1 epoch.
    def create_model_and_train(self, network, dataset, amp_level="O0", dataset_sink_mode=True,
                               loss=None, opt=None):
        model = self.create_train_model(network=network, amp_level=amp_level, loss=loss, opt=opt)

        logger.info(f"MindSporeTest::call model train with dataset_sink_mode={dataset_sink_mode}")
        model.train(epoch=1, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode)
        return True

    def create_parallel_model_and_predict(self, network, ckpt_name_list, *predict_data):
        model_predict = Model(network=network)
        predict_map = model_predict.infer_predict_layout(*predict_data)
        load_distributed_checkpoint(network, ckpt_name_list, predict_map)
        output = model_predict.predict(*predict_data)
        return output.asnumpy()

    def create_parallel_model_retrain_and_predict(self, network, dataset, ckpt_name_list,
                                                  *predict_data, amp_level="O0",
                                                  dataset_sink_mode=True, loss=None, opt=None):
        model_train = self.create_train_model(network, amp_level, loss, opt)
        train_map = model_train.infer_train_layout(dataset, dataset_sink_mode=dataset_sink_mode)
        load_distributed_checkpoint(model_train.train_network, ckpt_name_list, train_map)
        model_train.train(epoch=1, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode)
        context.set_auto_parallel_context(full_batch=True)
        output = model_train.predict(*predict_data)
        return output.asnumpy()


modeltrainbase = ModelTrainBase()
