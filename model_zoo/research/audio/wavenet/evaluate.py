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
"""evaluation"""
import os
from os.path import join
import argparse
import glob
import numpy as np
from scipy.io import wavfile
from hparams import hparams, hparams_debug_string
import audio
from tqdm import tqdm
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset.engine as de
from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset
from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_scalar_input
from src.dataset import RawAudioDataSource, MelSpecDataSource, DualDataset

parser = argparse.ArgumentParser(description='TTS training')
parser.add_argument('--data_path', type=str, required=True, default='',
                    help='Directory contains preprocessed features.')
parser.add_argument('--preset', type=str, required=True, default='', help='Path of preset parameters (json).')
parser.add_argument('--pretrain_ckpt', type=str, default='', help='Pretrained checkpoint path')
parser.add_argument('--is_numpy', action="store_true", default=False, help='Using numpy for inference or not')
parser.add_argument('--output_path', type=str, default='./out_wave/', help='Path to save generated audios')
parser.add_argument('--speaker_id', type=str, default='',
                    help=' Use specific speaker of data in case for multi-speaker datasets.')
parser.add_argument('--platform', type=str, default='GPU', choices=('GPU', 'CPU'),
                    help='run platform, support GPU and CPU. Default: GPU')
args = parser.parse_args()


def get_data_loader(hparam, data_dir):
    """
    test data loader
    """
    wav_paths = glob.glob(os.path.join(data_dir, "*-wave.npy"))
    if wav_paths:
        X = FileSourceDataset(RawAudioDataSource(data_dir,
                                                 hop_size=audio.get_hop_size(),
                                                 max_steps=None, cin_pad=hparam.cin_pad))
    else:
        X = None
    C = FileSourceDataset(MelSpecDataSource(data_dir,
                                            hop_size=audio.get_hop_size(),
                                            max_steps=None, cin_pad=hparam.cin_pad))

    length_x = np.array(C.file_data_source.lengths)
    if C[0].shape[-1] != hparam.cin_channels:
        raise RuntimeError("Invalid cin_channnels {}. Expected to be {}.".format(hparam.cin_channels, C[0].shape[-1]))

    dataset = DualDataset(X, C, length_x, batch_size=hparam.batch_size, hparams=hparam)

    data_loader = de.GeneratorDataset(dataset, ["x_batch", "y_batch", "c_batch", "g_batch", "input_lengths", "mask"])

    return data_loader, dataset


def batch_wavegen(hparam, net, c_input=None, g_input=None, tqdm_=None, is_numpy=True):
    """
    generate audio
    """
    assert c_input is not None
    B = c_input.shape[0]
    net.set_train(False)

    if hparam.upsample_conditional_features:
        length = (c_input.shape[-1] - hparam.cin_pad * 2) * audio.get_hop_size()
    else:
        # already dupulicated
        length = c_input.shape[-1]

    y_hat = net.incremental_forward(c=c_input, g=g_input, T=length, tqdm=tqdm_, softmax=True, quantize=True,
                                    log_scale_min=hparam.log_scale_min, is_numpy=is_numpy)

    if is_mulaw_quantize(hparam.input_type):
        # needs to be float since mulaw_inv returns in range of [-1, 1]
        y_hat = np.reshape(np.argmax(y_hat, 1), (B, -1))
        y_hat = y_hat.astype(np.float32)
        for k in range(B):
            y_hat[k] = P.inv_mulaw_quantize(y_hat[k], hparam.quantize_channels - 1)
    elif is_mulaw(hparam.input_type):
        y_hat = np.reshape(y_hat, (B, -1))
        for k in range(B):
            y_hat[k] = P.inv_mulaw(y_hat[k], hparam.quantize_channels - 1)
    else:
        y_hat = np.reshape(y_hat, (B, -1))

    if hparam.postprocess is not None and hparam.postprocess not in ["", "none"]:
        for k in range(B):
            y_hat[k] = getattr(audio, hparam.postprocess)(y_hat[k])

    if hparam.global_gain_scale > 0:
        for k in range(B):
            y_hat[k] /= hparam.global_gain_scale

    return y_hat


def to_int16(x_):
    """
    convert datatype to int16
    """
    if x_.dtype == np.int16:
        return x_
    assert x_.dtype == np.float32
    assert x_.min() >= -1 and x_.max() <= 1.0
    return (x_ * 32767).astype(np.int16)


def get_reference_file(hparam, dataset_source, idx):
    """
    get reference files
    """
    reference_files = []
    reference_feats = []
    for _ in range(hparam.batch_size):
        if hasattr(dataset_source, "X"):
            reference_files.append(dataset_source.X.collected_files[idx][0])
        else:
            pass
        if hasattr(dataset_source, "Mel"):
            reference_feats.append(dataset_source.Mel.collected_files[idx][0])
        else:
            reference_feats.append(dataset_source.collected_files[idx][0])
        idx += 1
    return reference_files, reference_feats, idx


def get_saved_audio_name(has_ref_file_, ref_file, ref_feat, g_fp):
    """get path to save reference audio"""
    if has_ref_file_:
        target_audio_path = ref_file
        name = os.path.splitext(os.path.basename(target_audio_path))[0].replace("-wave", "")
    else:
        target_feat_path = ref_feat
        name = os.path.splitext(os.path.basename(target_feat_path))[0].replace("-feats", "")
    # Paths
    if g_fp is None:
        dst_wav_path_ = join(args.output_path, "{}_gen.wav".format(name))
        target_wav_path_ = join(args.output_path, "{}_ref.wav".format(name))
    else:
        dst_wav_path_ = join(args.output_path, "speaker{}_{}_gen.wav".format(g, name))
        target_wav_path_ = join(args.output_path, "speaker{}_{}_ref.wav".format(g, name))
    return dst_wav_path_, target_wav_path_


def save_ref_audio(hparam, ref, length, target_wav_path_):
    """
    save reference audio
    """
    if is_mulaw_quantize(hparam.input_type):
        ref = np.reshape(np.argmax(ref, 0), (-1))[:length]
        ref = ref.astype(np.float32)
    else:
        ref = np.reshape(ref, (-1))[:length]

    if is_mulaw_quantize(hparam.input_type):
        ref = P.inv_mulaw_quantize(ref, hparam.quantize_channels - 1)
    elif is_mulaw(hparam.input_type):
        ref = P.inv_mulaw(ref, hparam.quantize_channels - 1)
    if hparam.postprocess is not None and hparam.postprocess not in ["", "none"]:
        ref = getattr(audio, hparam.postprocess)(ref)
    if hparam.global_gain_scale > 0:
        ref /= hparam.global_gain_scale

    ref = np.clip(ref, -1.0, 1.0)

    wavfile.write(target_wav_path_, hparam.sample_rate, to_int16(ref))


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=False)
    speaker_id = int(args.speaker_id) if args.speaker_id != '' else None
    if args.preset is not None:
        with open(args.preset) as f:
            hparams.parse_json(f.read())

    assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate
    hparams.batch_size = 10
    hparams.max_time_sec = None
    hparams.max_time_steps = None
    data_loaders, source_dataset = get_data_loader(hparam=hparams, data_dir=args.data_path)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
    )

    param_dict = load_checkpoint(args.pretrain_ckpt)
    load_param_into_net(model, param_dict)
    print('Successfully loading the pre-trained model')

    os.makedirs(args.output_path, exist_ok=True)
    cin_pad = hparams.cin_pad

    file_idx = 0
    for data in data_loaders.create_dict_iterator():
        x, y, c, g, input_lengths = data['x_batch'], data['y_batch'], data['c_batch'], data['g_batch'], data[
            'input_lengths']
        if cin_pad > 0:
            c = c.asnumpy()
            c = np.pad(c, pad_width=(cin_pad, cin_pad), mode="edge")
            c = Tensor(c)

        ref_files, ref_feats, file_idx = get_reference_file(hparams, source_dataset, file_idx)
        # Generate
        y_hats = batch_wavegen(hparams, model, data['c_batch'], tqdm_=tqdm, is_numpy=args.is_numpy)
        x = x.asnumpy()
        input_lengths = input_lengths.asnumpy()
        # Save each utt.
        has_ref_file = bool(ref_files)
        for i, (ref_, gen_, length_) in enumerate(zip(x, y_hats, input_lengths)):
            dst_wav_path, target_wav_path = get_saved_audio_name(has_ref_file_=has_ref_file, ref_file=ref_files[i],
                                                                 ref_feat=ref_feats[i], g_fp=g)
            save_ref_audio(hparams, ref_, length_, target_wav_path)

            gen = gen_[:length_]
            gen = np.clip(gen, -1.0, 1.0)
            wavfile.write(dst_wav_path, hparams.sample_rate, to_int16(gen))
