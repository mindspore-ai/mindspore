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
'''python dataset.py'''

import os
import argparse
import pandas as pd
import numpy as np
import librosa
from mindspore.mindrecord import FileWriter
from mindspore import context
from src.config import data_cfg as cfg


def compute_melgram(audio_path, save_path='', filename='', save_npy=True):
    """
    extract melgram feature from the audio and save as numpy array

    Args:
        audio_path (str): path to the audio clip.
        save_path (str): path to save the numpy array.
        filename (str): filename of the audio clip.

    Returns:
        numpy array.

    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, _ = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) //
                  2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(
        melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[np.newaxis, np.newaxis, :]
    if save_npy:

        save_path = save_path + filename[:-4] + '.npy'
        np.save(save_path, ret)
    return ret


def get_data(features_data, labels_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i, "feature": feature, "label": label}
        data_list.append(data_json)
    return data_list


def convert(s):
    if s.isdigit():
        return int(s)
    return s


def GetLabel(info_path, info_name):
    """
    separate dataset into training set and validation set

    Args:
        info_path (str): path to the information file.
        info_name (str): name of the information file.

    """
    T = []
    with open(info_path + '/' + info_name, 'rb') as info:
        data = info.readline()
        while data:
            T.append([
                convert(i[1:-1])
                for i in data.strip().decode('utf-8').split("\t")
            ])
            data = info.readline()

    annotation = pd.DataFrame(T[1:], columns=T[0])
    count = []
    for i in annotation.columns[1:-2]:
        count.append([annotation[i].sum() / len(annotation), i])
    count = sorted(count)
    full_label = []
    for i in count[-50:]:
        full_label.append(i[1])
    out = []
    for i in T[1:]:
        index = [k for k, x in enumerate(i) if x == 1]
        label = [T[0][k] for k in index]
        L = [str(0) for k in range(50)]
        L.append(i[-1])
        for j in label:
            if j in full_label:
                ind = full_label.index(j)
                L[ind] = '1'
        out.append(L)
    out = np.array(out)

    Train = []
    Val = []

    for i in out:
        if np.random.rand() > 0.2:
            Train.append(i)
        else:
            Val.append(i)
    np.savetxt("{}/music_tagging_train_tmp.csv".format(info_path),
               np.array(Train),
               fmt='%s',
               delimiter=',')
    np.savetxt("{}/music_tagging_val_tmp.csv".format(info_path),
               np.array(Val),
               fmt='%s',
               delimiter=',')


def generator_md(info_name, file_path, num_classes):
    """
    generate numpy array from features of all audio clips

    Args:
        info_path (str): path to the information file.
        file_path (str): path to the npy files.

    Returns:
        2 numpy array.

    """
    df = pd.read_csv(info_name, header=None)
    df.columns = [str(i) for i in range(num_classes)] + ["mp3_path"]
    data = []
    label = []
    for i in range(len(df)):
        try:
            data.append(
                np.load(file_path + df.mp3_path.values[i][:-4] +
                        '.npy').reshape(1, 96, 1366))
            label.append(np.array(df[df.columns[:-1]][i:i + 1])[0])
        except FileNotFoundError:
            print("Exception occurred in generator_md.")
    return np.array(data), np.array(label, dtype=np.int32)


def convert_to_mindrecord(info_name, file_path, store_path, mr_name,
                          num_classes):
    """ convert dataset to mindrecord """
    num_shard = 4
    data, label = generator_md(info_name, file_path, num_classes)
    schema_json = {
        "id": {
            "type": "int32"
        },
        "feature": {
            "type": "float32",
            "shape": [1, 96, 1366]
        },
        "label": {
            "type": "int32",
            "shape": [num_classes]
        }
    }

    writer = FileWriter(
        os.path.join(store_path, '{}.mindrecord'.format(mr_name)), num_shard)
    datax = get_data(data, label)
    writer.add_schema(schema_json, "music_tagger_schema")
    writer.add_index(["id"])
    writer.write_raw_data(datax)
    writer.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get feature')
    parser.add_argument('--device_id',
                        type=int,
                        help='device ID',
                        default=None)
    args = parser.parse_args()

    if cfg.get_npy:
        GetLabel(cfg.info_path, cfg.info_name)
        dirname = os.listdir(cfg.audio_path)
        for d in dirname:
            file_name = os.listdir("{}/{}".format(cfg.audio_path, d))
            if not os.path.isdir("{}/{}".format(cfg.npy_path, d)):
                os.mkdir("{}/{}".format(cfg.npy_path, d))
            for f in file_name:
                compute_melgram("{}/{}/{}".format(cfg.audio_path, d, f),
                                "{}/{}/".format(cfg.npy_path, d), f)

    if cfg.get_mindrecord:
        if args.device_id is not None:
            context.set_context(device_target='Ascend',
                                mode=context.GRAPH_MODE,
                                device_id=args.device_id)
        else:
            context.set_context(device_target='Ascend',
                                mode=context.GRAPH_MODE,
                                device_id=cfg.device_id)
        for cmn in cfg.mr_nam:
            if cmn in ['train', 'val']:
                convert_to_mindrecord('music_tagging_{}_tmp.csv'.format(cmn),
                                      cfg.npy_path, cfg.mr_path, cmn,
                                      cfg.num_classes)
