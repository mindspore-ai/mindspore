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
"""Transformer evaluation script."""
import os
import time
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from src.dataset import create_gru_dataset
from src.seq2seq import Seq2Seq
from src.gru_for_infer import GRUInferCell

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.output_file = os.path.join(config.output_path, config.output_file)
    config.target_file = os.path.join(config.output_path, config.target_file)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_gru_eval():
    """
    Transformer evaluation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, reserve_class_name_in_scope=False,
                        device_id=get_device_id(), save_graphs=False)
    mindrecord_file = config.dataset_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.eval_batch_size,
                                 dataset_path=mindrecord_file, rank_size=get_device_num(), rank_id=0,
                                 do_shuffle=False, is_training=False)
    dataset_size = dataset.get_dataset_size()
    print("dataset size is {}".format(dataset_size))
    network = Seq2Seq(config, is_training=False)
    network = GRUInferCell(network)
    network.set_train(False)
    if config.ckpt_file != "":
        parameter_dict = load_checkpoint(config.ckpt_file)
        load_param_into_net(network, parameter_dict)
    model = Model(network)

    predictions = []
    source_sents = []
    target_sents = []
    eval_text_len = 0
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_ids"])
        target_sents.append(batch["target_ids"])
        source_ids = Tensor(batch["source_ids"], mstype.int32)
        target_ids = Tensor(batch["target_ids"], mstype.int32)
        predicted_ids = model.predict(source_ids, target_ids)
        print("predicts is ", predicted_ids.asnumpy())
        print("target_ids is ", target_ids)
        predictions.append(predicted_ids.asnumpy())
        eval_text_len = eval_text_len + 1

    f_output = open(config.output_file, 'w')
    f_target = open(config.target_file, "w")
    for batch_out, true_sentence in zip(predictions, target_sents):
        for i in range(config.eval_batch_size):
            target_ids = [str(x) for x in true_sentence[i].tolist()]
            f_target.write(" ".join(target_ids) + "\n")
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f_output.write(" ".join(token_ids) + "\n")
    f_output.close()
    f_target.close()

if __name__ == "__main__":
    run_gru_eval()
