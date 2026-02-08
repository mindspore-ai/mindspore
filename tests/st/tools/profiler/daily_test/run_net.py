# Copyright 2022 Huawei Technologies Co., Ltd
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
"""run net"""
from argparse import ArgumentParser

from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset


def train_with_profiler():
    """Train Net with profiling."""
    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
    for src, tgt in nlp_dataset:
        net(src, tgt)


parser = ArgumentParser(description='test env enable profiler')
parser.add_argument('--target', type=str)
parser.add_argument('--mode', type=int)
args = parser.parse_args()
train_with_profiler()
