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
"""The utils of model testing."""
import os
import sys
import math
import jieba
import numpy as np

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, os.path.join(workspace, "mindformers"))

from mindformers.tools.logger import logger


def _get_all_words(standard_cut_infer_ret_list, test_cut_infer_ret_list):
    all_words = []
    for s_cut in standard_cut_infer_ret_list:
        if s_cut not in all_words:
            all_words.append(s_cut)
    for t_cut in test_cut_infer_ret_list:
        if t_cut not in all_words:
            all_words.append(t_cut)
    return all_words


def _get_word_vector(standard_cut_infer_ret_list, test_cut_infer_ret_list, all_words):
    la_standard = []
    lb_test = []
    for word in all_words:
        la_standard.append(standard_cut_infer_ret_list.count(word))
        lb_test.append(test_cut_infer_ret_list.count(word))
    return la_standard, lb_test


def _get_calculate_cos(la_standard, lb_test):
    laa = np.array(la_standard)
    lbb = np.array(lb_test)
    cos = (np.dot(laa, lbb.T)) / ((math.sqrt(np.dot(laa, laa.T))) * (math.sqrt(np.dot(lbb, lbb.T))))
    return np.round(cos, 2)


def compare_distance(x1, x2, bench_sim=0.95):
    """compare distance"""
    y1 = list(jieba.cut(x1))
    y2 = list(jieba.cut(x2))
    all_words = _get_all_words(y1, y2)
    laa, lbb = _get_word_vector(y1, y2, all_words)
    sim = _get_calculate_cos(laa, lbb)
    logger.info("calculate sim is:{}".format(str(sim)))
    assert sim >= bench_sim
