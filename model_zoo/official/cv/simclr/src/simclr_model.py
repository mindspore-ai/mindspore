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
"""SimCLR Model class."""
from mindspore import nn
from .resnet import _fc

class Identity(nn.Cell):
    def construct(self, x):
        return x

class SimCLR(nn.Cell):
    """
    SimCLR Model.
    """
    def __init__(self, encoder, project_dim, n_features):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.encoder.end_point = Identity()
        self.dense1 = _fc(self.n_features, self.n_features)
        self.relu = nn.ReLU()
        self.end_point = _fc(self.n_features, project_dim)

    # Projector MLP.
    def projector(self, x):
        out = self.dense1(x)
        out = self.relu(out)
        out = self.end_point(out)
        return out

    def construct(self, x_i, x_j):
        h_i = self.encoder(x_i)
        z_i = self.projector(h_i)

        h_j = self.encoder(x_j)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

    def inference(self, x):
        h = self.encoder(x)
        return h

class SimCLR_Classifier(nn.Cell):
    """
    SimCLR with Classifier.
    """
    def __init__(self, encoder, classifier):
        super(SimCLR_Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.softmax = nn.Softmax()

    def construct(self, x):
        y = self.encoder(x)
        z = self.classifier(y)
        return self.softmax(z)
