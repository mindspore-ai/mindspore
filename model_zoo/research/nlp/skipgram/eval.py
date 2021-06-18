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
"""
eval skipgram according to model file:
python eval.py --checkpoint_path=[CHECKPOINT_PATH] --dictionary=[ID2WORD_DICTIONARY] &> eval.log &
"""

import os
import argparse
import numpy as np

from mindspore.train.serialization import load_param_into_net, load_checkpoint

from src.dataset import load_eval_data
from src.config import w2v_cfg
from src.utils import cal_top_k_similar, get_w2v_emb
from src.skipgram import SkipGram

parser = argparse.ArgumentParser(description='Evaluate SkipGram')
parser.add_argument('--checkpoint_path', type=str, default=None, help='checkpoint file path.')
parser.add_argument('--dictionary', type=str, default=None, help='map word\'s identity number to word.')
parser.add_argument('--eval_data_dir', type=str, default=None, help='evaluation file\'s direcionary.')
args = parser.parse_args()

if __name__ == '__main__':
    if args.checkpoint_path is not None and args.dictionary is not None:
        id2word = np.load(args.dictionary, allow_pickle=True).item()  # dict()
        net = SkipGram(len(id2word), w2v_cfg.emb_size)
        load_param_into_net(net, load_checkpoint(args.checkpoint_path))
        w2v_emb = get_w2v_emb(net, id2word)
    else:
        w2v_emb = np.load(os.path.join(w2v_cfg.w2v_emb_save_dir, 'w2v_emb.npy'), allow_pickle=True).item()  # dict()
    if args.eval_data_dir is not None:
        samples = load_eval_data(args.eval_data_dir)
    else:
        samples = load_eval_data(w2v_cfg.eval_data_dir)  # dict(): {question type: sample list}
    emb_list = list(w2v_emb.items())
    emb_matrix = np.array([item[1] for item in emb_list])  # vocab_size * emb_size
    target_embs = []
    labels = []
    ignores = []
    for sample_type in samples:
        type_k = samples[sample_type]
        for sample in type_k:
            try:
                vecs = [w2v_emb[w] for w in sample]
            except KeyError:
                continue
            vecs = [vec / np.linalg.norm(vec) for vec in vecs]  # l2-normalize
            target_embs.append((vecs[1] + vecs[2] - vecs[0]) / 3)  # average
            labels.append(sample[3])
            ignores.append([sample[0], sample[1], sample[2]])
    top_k_similar = cal_top_k_similar(np.array(target_embs), emb_matrix, k=5)

    correct_cnt = 0
    for i, candidate_index in enumerate(top_k_similar):
        ignore = ignores[i]
        label = labels[i]
        for k in candidate_index:
            predicted = emb_list[k][0]
            if predicted not in ignore:  # Similar to gensim, ignore the word in the 'example' here.
                break
        if predicted == label:
            correct_cnt += 1
        print('predicted: %-15s label: %s'% (predicted, label))
    print("Total Accuracy: %.2f%%"% (correct_cnt / len(target_embs) * 100))
