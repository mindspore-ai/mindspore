# Copyright 2023 Huawei Technologies Co., Ltd
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
Test LiteInfer python API.
"""
import os
import pytest
import mindspore_lite as mslite

# if import error or env variable 'MSLITE_ENABLE_CLOUD_INFERENCE' is not set, pass all tests
env_ok = True

if os.environ.get('MSLITE_ENABLE_CLOUD_INFERENCE') != 'on':
    print("================== test_lite_infer_api MSLITE_ENABLE_CLOUD_INFERENCE not on")
    env_ok = False


# ============================ LiteInfer ============================
def test_lite_llm_engine_llm_req_parameter_type_check():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.decoder_cluster_id = "123"
    assert "decoder_cluster_id must be int, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_parameter_type_check():
    if not env_ok:
        return
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0)
        llm_engine.init("model.mindir", {})
    assert "model_paths must be tuple/list of str, but got" in str(raise_info.value)
