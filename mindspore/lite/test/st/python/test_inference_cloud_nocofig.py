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
"""
Test lite python API.
"""
import sys
import os
import traceback
from functools import wraps
import numpy as np
import mindspore_lite as mslite


def lite_test(func):
    @wraps(func)
    def wrap_test(*args, **kwargs):
        try:
            print(f"[BEGIN] {str(func.__name__)}")
            func(*args, **kwargs)
            print(f"[SUCCESS] {str(func.__name__)}")
        except Exception as e:
            traceback.print_exc()
            print(f"[FAILED] {str(func.__name__)}")
            raise
    return wrap_test


# ============================ ascend inference ============================
@lite_test
def test_model_group_inference_ascend(mindir_dir):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    context.ascend.provider = "ge"
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
    model0 = mslite.Model()
    model1 = mslite.Model()
    model_group.add_model([model0, model1])

    model_path0 = os.path.join(mindir_dir, "model_group_first.mindir")
    model_path1 = os.path.join(mindir_dir, "model_group_second.mindir")

    model0.build_from_file(model_path0, mslite.ModelType.MINDIR, context)
    model1.build_from_file(model_path1, mslite.ModelType.MINDIR, context)

    for i in range(2):
        inputs = [mslite.Tensor(np.ones((4, 4), np.float32))]
        outputs = model0.predict(inputs)
        assert (outputs[0].get_data_to_numpy() == np.ones((4, 4), np.float32)).all()

        inputs = [mslite.Tensor(np.ones((4, 1), np.float32))]
        outputs = model1.predict(inputs)
        assert (outputs[0].get_data_to_numpy() == (np.ones((4, 4), np.float32) * 2)).all()


if __name__ == '__main__':
    print("test_inference_cloud_nocofig.py: begin run testcases.")
    model_dir = sys.argv[1]
    backend = sys.argv[2]
    if backend == "Ascend":
        test_model_group_inference_ascend(model_dir)
    else:
        print(f'test_inference_cloud_nocofig.py: skip backend {backend}!')
    print("test_inference_cloud_nocofig.py: run testcases success.")
