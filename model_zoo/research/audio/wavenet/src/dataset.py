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
"""
Create train dataset.
"""
import os
import math
import numpy as np
import audio
from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii import preprocessing as P
from wavenet_vocoder.util import is_mulaw_quantize
from train_pytorch import _pad, _pad_2d, to_categorical, ensure_divisible, RawAudioDataSource, MelSpecDataSource, assert_ready_for_upsampling
import mindspore.dataset.engine as de


def sequence_mask(sequence_length, max_len=None):
    """make sequence mask for loss"""
    if max_len is None:
        max_len = np.max(sequence_length)
    batch_size = len(sequence_length)
    seq_range = np.linspace(0, max_len - 1, max_len, dtype=np.int32)
    seq_range_expand = np.tile(np.expand_dims(seq_range, 0), (batch_size, 1))
    seq_length_expand = np.tile(np.expand_dims(sequence_length, 1), (1, max_len))
    seq_length_expand = np.expand_dims(np.array(seq_range_expand < seq_length_expand, dtype=np.float32), -1)
    return seq_length_expand

class DistributedSampler():
    """function to distribute and shuffle sample
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(self.dataset)
        self.num_samplers = int(math.ceil(self.dataset_len * 1.0 / self.group_size))
        self.total_size = self.num_samplers * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(self.dataset_len))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samplers


def process_condition_batch(max_time_steps, hparams, batch):
    """process condition batch"""
    cin_pad = hparams.cin_pad
    new_batch = []
    for batch_ in batch:
        x, c, g = batch_
        if hparams.upsample_conditional_features:
            assert_ready_for_upsampling(x, c, cin_pad=0)
            if max_time_steps is not None:
                max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                if len(x) > max_steps:
                    max_time_frames = max_steps // audio.get_hop_size()
                    s = np.random.randint(cin_pad, len(c) - max_time_frames - cin_pad)
                    ts = s * audio.get_hop_size()
                    x = x[ts:ts + audio.get_hop_size() * max_time_frames]
                    c = c[s - cin_pad:s + max_time_frames + cin_pad, :]
                    assert_ready_for_upsampling(x, c, cin_pad=cin_pad)
        else:
            x, c = audio.adjust_time_resolution(x, c)
            if max_time_steps is not None and len(x) > max_time_steps:
                s = np.random.randint(cin_pad, len(x) - max_time_steps - cin_pad)
                x = x[s:s + max_time_steps]
                c = c[s - cin_pad:s + max_time_steps + cin_pad, :]
            assert len(x) == len(c)
        new_batch.append((x, c, g))
    return new_batch


def process_no_condition_batch(max_time_steps, batch):
    """process no condition batch"""
    new_batch = []
    for batch_ in batch:
        x, c, g = batch_
        x = audio.trim(x)
        if max_time_steps is not None and len(x) > max_time_steps:
            s = np.random.randint(0, len(x) - max_time_steps)
            x = x[s:s + max_time_steps]
        new_batch.append((x, c, g))
    return new_batch



def collate_fn(batch, hparams):
    """
    Create batch
    """
    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.gin_channels > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None

    if local_conditioning:
        new_batch = process_condition_batch(max_time_steps, hparams, batch)
    else:
        new_batch = process_no_condition_batch(max_time_steps, batch)
    batch = new_batch
    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)
    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        x_batch = np.array(
            [_pad_2d(to_categorical(x[0], num_classes=hparams.quantize_channels), max_input_len, 0, padding_value) for x
             in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        y_batch = np.array([_pad(x[0], max_input_len, constant_values=padding_value)
                            for x in batch], dtype=np.int32)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = c_batch.transpose((0, 2, 1))
    else:
        c_batch = np.zeros(hparams.batch_size, dtype=np.float32)

    if global_conditioning:
        g_batch = [x[2] for x in batch]
    else:
        # g_batch = None # MindSpore does not support None input
        g_batch = np.zeros(hparams.batch_size, dtype=np.int64)

    # Convert to channel first (B, C, T)
    x_batch = x_batch.transpose((0, 2, 1))
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = np.expand_dims(y_batch, axis=-1)
    else:
        y_batch = np.expand_dims(y_batch, axis=-1)

    input_lengths = input_lengths

    mask = sequence_mask(input_lengths, max_len=x_batch.shape[-1])

    return x_batch, y_batch, c_batch, g_batch, input_lengths, mask


class DualDataset():
    """Create Dataset loader for audio Mel and Audio"""
    def __init__(self, X, Mel, length, batch_size, hparams):
        self.multi_speaker = X.file_data_source.multi_speaker
        self.X = X
        self.Mel = Mel
        self.length = length
        self.hparams = hparams
        self.sorted_index = list(np.argsort(length))
        self.bins = [self.sorted_index[i:i + batch_size] for i in range(0, len(self.sorted_index), batch_size)]
        if len(self.sorted_index) / batch_size != 0:
            self.bins.append(self.sorted_index[-batch_size:])
        self.size = len(self.bins)

    def __getitem__(self, idx):
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        combined_data = []
        mel_len, audio_len = [], []
        for i in self.bins[idx]:
            if self.Mel is not None:
                mel = self.Mel[i]
                raw_audio = self.X[i]
                length_mel, length_x = mel.shape[0], raw_audio.shape[0]
                combined_data.append((raw_audio, mel, speaker_id))
                mel_len.append(length_mel)
                audio_len.append(length_x)
            else:
                raw_audio = self.X[i]
                length_x = raw_audio.shape[0]
                combined_data.append((raw_audio, speaker_id))
                audio_len.append(length_x)

        x_batch, y_batch, c_batch, g_batch, input_lengths, mask = collate_fn(combined_data, self.hparams)

        return x_batch, y_batch, c_batch, g_batch, input_lengths, mask

    def __len__(self):
        return self.size


def get_data_loaders(dump_root, speaker_id, hparams=None, rank_id=None, group_size=None):
    """create train dataset"""
    local_conditioning = hparams.cin_channels > 0

    if hparams.max_time_steps is not None:
        max_steps = ensure_divisible(hparams.max_time_steps, audio.get_hop_size(), True)
    else:
        max_steps = None

    X = FileSourceDataset(
        RawAudioDataSource(os.path.join(dump_root, 'train_no_dev'), speaker_id=speaker_id,
                           max_steps=max_steps, cin_pad=hparams.cin_pad,
                           hop_size=audio.get_hop_size()))

    if local_conditioning:
        Mel = FileSourceDataset(
            MelSpecDataSource(os.path.join(dump_root, 'train_no_dev'), speaker_id=speaker_id,
                              max_steps=max_steps, cin_pad=hparams.cin_pad,
                              hop_size=audio.get_hop_size()))
        assert len(X) == len(Mel)
        print("Local conditioning enabled. Shape of a sample: {}.".format(Mel[0].shape))
    else:
        Mel = None
    print("length of the dataset is {}".format(len(X)))
    length_x = np.array(X.file_data_source.lengths)
    dataset = DualDataset(X, Mel, length_x, batch_size=hparams.batch_size, hparams=hparams)
    sampler = DistributedSampler(dataset, rank_id, group_size, shuffle=True, seed=0)
    data_loaders = de.GeneratorDataset(dataset, ["x_batch", "y_batch", "c_batch", "g_batch", "input_lengths", "mask"],
                                       sampler=sampler)

    return data_loaders
