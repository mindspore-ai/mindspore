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
# ===========================================================================
"""Download process data."""
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
from glob import glob
import logging
from six.moves import urllib
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from tqdm import tqdm
from utils import prepare_words_list
from src.model_utils.config import config, prepare_model_settings


FLAGS = None
MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
K = 0


def which_set(filename, validation_percentage, testing_percentage):
    '''Which set.'''
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(bytes(hash_name, 'utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

class AudioProcessor():
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
                 wanted_words, validation_percentage, testing_percentage,
                 model_settings):
        self.data_dir = data_dir
        self.maybe_download_and_extract_dataset(data_url, data_dir)
        self.prepare_data_index(silence_percentage, unknown_percentage,
                                wanted_words, validation_percentage,
                                testing_percentage)
        self.prepare_background_data()
        self.prepare_data(model_settings)

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        '''Maybe download and extract dataset.'''
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                logging.error('Failed to download URL: %s to folder: %s', data_url,
                              filepath)
                logging.error('Please make sure you have enough free space and'
                              ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            logging.info('Successfully downloaded %s (%d bytes)', filename,
                         statinfo.st_size)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                           wanted_words, validation_percentage,
                           testing_percentage):
        '''Prepare data index.'''
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        '''Prepare background data.'''
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data

        search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                   '*.wav')
        for wav_path in glob(search_path):
            wav_data, _ = sf.read(wav_path)
            self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in ' + search_path)
        return None

    def prepare_single_sample(self, wav_filename, foreground_volume, time_shift_padding, time_shift_offset,
                              desired_samples, background_data, background_volume):
        '''Prepare single sample.'''
        wav_data, _ = sf.read(wav_filename)
        if len(wav_data) < desired_samples:
            wav_data = np.pad(wav_data, [0, desired_samples - len(wav_data)], 'constant')
        scaled_foreground = wav_data * foreground_volume
        padded_foreground = np.pad(scaled_foreground, time_shift_padding, 'constant')
        sliced_foreground = padded_foreground[time_shift_offset: time_shift_offset + desired_samples]
        background_add = background_data[0] * background_volume + sliced_foreground
        background_clamp = np.clip(background_add, -1.0, 1.0)
        feature = mfcc(background_clamp, samplerate=config.sample_rate, winlen=config.window_size_ms / 1000,
                       winstep=config.window_stride_ms / 1000,
                       numcep=config.dct_coefficient_count, nfilt=40, nfft=1024, lowfreq=20, highfreq=7000).flatten()
        return feature

    def prepare_data(self, model_settings):
        '''Prepare data.'''
        # Pick one of the partitions to choose samples from.
        time_shift = int((config.time_shift_ms * config.sample_rate) / 1000)
        background_frequency = config.background_frequency
        background_volume_range = config.background_volume
        desired_samples = model_settings['desired_samples']
        if not os.path.exists(config.download_feat_dir):
            os.makedirs(config.download_feat_dir, exist_ok=True)
        for mode in ['training', 'validation', 'testing']:
            candidates = self.data_index[mode]
            sample_count = len(candidates)
            # Data and labels will be populated and returned.
            data = np.zeros((sample_count, model_settings['fingerprint_size']))
            labels = np.zeros(sample_count)
            use_background = self.background_data and (mode == 'training')
            for i in tqdm(range(sample_count)):
                # Pick which audio sample to use.
                sample_index = i
                sample = candidates[sample_index]
                # If we're time shifting, set up the offset for this sample.
                if time_shift > 0:
                    time_shift_amount = np.random.randint(-time_shift, time_shift)
                else:
                    time_shift_amount = 0
                if time_shift_amount > 0:
                    time_shift_padding = [[time_shift_amount, 0]]
                    time_shift_offset = 0
                else:
                    time_shift_padding = [[0, -time_shift_amount]]
                    time_shift_offset = -time_shift_amount
                if use_background:
                    background_index = np.random.randint(len(self.background_data))
                    background_samples = self.background_data[background_index]
                    background_offset = np.random.randint(
                        0, len(background_samples) - model_settings['desired_samples'])
                    background_clipped = background_samples[background_offset:(
                        background_offset + desired_samples)]
                    background_reshaped = background_clipped.reshape([desired_samples, 1])
                    if np.random.uniform(0, 1) < background_frequency:
                        background_volume = np.random.uniform(0, background_volume_range)
                    else:
                        background_volume = 0
                else:
                    background_reshaped = np.zeros([desired_samples, 1])
                    background_volume = 0
                if sample['label'] == SILENCE_LABEL:
                    foreground_volume = 0
                else:
                    foreground_volume = 1
                data[i, :] = self.prepare_single_sample(sample['file'], foreground_volume, time_shift_padding,
                                                        time_shift_offset, desired_samples,
                                                        background_reshaped, background_volume)
                label_index = self.word_to_index[sample['label']]
                labels[i] = label_index
            np.save(os.path.join(config.download_feat_dir, '{}_data.npy'.format(mode)), data)
            np.save(os.path.join(config.download_feat_dir, '{}_label.npy'.format(mode)), labels)


if __name__ == '__main__':
    print('start download_process')
    model_settings_1 = prepare_model_settings(
        len(prepare_words_list(config.wanted_words.split(','))),
        config.sample_rate, config.clip_duration_ms, config.window_size_ms,
        config.window_stride_ms, config.dct_coefficient_count)
    audio_processor = AudioProcessor(
        config.download_data_url, config.data_dir, config.silence_percentage,
        config.unknown_percentage,
        config.wanted_words.split(','), config.validation_percentage,
        config.testing_percentage, model_settings_1)
