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
""" modeltrain management base module. """
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.serialization import load_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train import Model
from ._utils import clean_all_ckpt_files
from ._utils import find_newest_ckpt_file
from mindspore import log as logger


class ModelTrainBase:
    def __init__(self):
        pass

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

    def load_newest_ckpt_from_model_train(self, model, epoch, dataset, callback=None,
                                        dataset_sink_mode=True,
                                        ckpt_path="./", ckpt_prefix="ckpt_ms", async_save=False,
                                        save_checkpoint_steps=1, sink_size=-1,
                                        integrated_save=True,
                                        load_format="default"):
        logger.info("MindSporeTest::configure Config to save Checkpoint")
        ckpt_config = CheckpointConfig(keep_checkpoint_max=5, integrated_save=integrated_save,
                                    save_checkpoint_steps=save_checkpoint_steps,
                                    async_save=async_save)
        ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
        logger.info(f"MindSporeTest::clean all Checkpoint file under {ckpt_path}")
        clean_all_ckpt_files(ckpt_path)
        callbacks = [ckpt_callback]
        if callback is not None and isinstance(callback, Callback):
            if isinstance(callback, list):
                callbacks.extend([cb for cb in callback if isinstance(cb, Callback)])
            else:
                callbacks.append(callback)
        logger.info(f"MindSporeTest::Model train and save checkpoint under {ckpt_path}")
        model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                    callbacks=callbacks, sink_size=sink_size)
        logger.info("MindSporeTest::load the newest checkpoint file and return")
        newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
        return load_checkpoint(newest_ckpt_file)


modeltrainbase = ModelTrainBase()
