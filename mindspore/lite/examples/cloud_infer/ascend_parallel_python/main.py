# Copyright 2024 Huawei Technologies Co., Ltd
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
This example is based on Stable Diffusion 2.1.
Model (batch_size = 1) is copied to 2 devices.
If run on 300IDUO, the devices above are two different chips in one NPU.
"""

import time
from parallel_predict_utils import ParallelPredictUtils
import numpy as np

SAMPLE_SIZE = [2, 4, 64, 64]
TIMESTEP_VALUE = 50
ENCODER_HIDDEN_STATES_SIZE = [2, 77, 1024]
DEVICE_NUM = 2

mindir_path = "SD2.1_graph.mindir"
sample = np.random.uniform(size=SAMPLE_SIZE).astype(np.float32)
timestep = np.array(TIMESTEP_VALUE, np.float32)
encoder_hidden_states = np.random.uniform(size=ENCODER_HIDDEN_STATES_SIZE).astype(np.float32)

# data slice
input_data_0 = [sample[0], timestep, encoder_hidden_states[0]]
input_data_1 = [sample[1], timestep, encoder_hidden_states[1]]
input_data_all = [input_data_0, input_data_1]

mslite_predict_utils = ParallelPredictUtils(mindir_path, DEVICE_NUM)
mslite_predict_utils.build_model()

t0 = time.time()
predict_result = mslite_predict_utils.run_predict(input_data_all)
print("Total predict time = ", time.time() - t0)
mslite_predict_utils.finalize()

print("=========== success ===========")
