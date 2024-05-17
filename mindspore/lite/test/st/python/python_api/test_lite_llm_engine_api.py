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
import sys
import socket
import numpy as np
import pytest
import mindspore_lite as mslite


# ============================ LLMClusterInfo ============================
def test_lite_llm_engine_cluster_info_role_type_check():
    with pytest.raises(TypeError) as raise_info:
        _ = mslite.LLMClusterInfo("abc", 0)
    assert "remote_role must be LLMRole, but got" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        llm_cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
        llm_cluster.remote_role = "abc"
    assert "remote_role must be LLMRole, but got" in str(raise_info.value)


def test_lite_llm_engine_cluster_info_cluster_id_type_check():
    with pytest.raises(TypeError) as raise_info:
        _ = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, "0")
    assert "remote_cluster_id must be int, but got" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        llm_cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
        llm_cluster.remote_cluster_id = "0"
    assert "remote_cluster_id must be int, but got" in str(raise_info.value)


def test_lite_llm_engine_cluster_info_address_type_check():
    with pytest.raises(TypeError) as raise_info:
        llm_cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
        llm_cluster.append_remote_ip_info((1.1, 2046))
    assert "address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
        llm_cluster.append_local_ip_info((1.1, 2046))
    assert "address must be in format of ('xxx.xxx.xxx.xxx', xxx) or (xxx, xxx), but got" in str(raise_info.value)
    llm_cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
    llm_cluster.append_local_ip_info(("192.168.0.1", 2046))
    local_infos = llm_cluster.local_ip_infos
    assert isinstance(local_infos, (tuple, list)) and len(local_infos) == 1
    assert isinstance(local_infos[0], (tuple, list)) and len(local_infos[0]) == 2

    expect_ip = socket.inet_aton("192.168.0.1")
    expect_ip = int.from_bytes(expect_ip, byteorder=sys.byteorder)
    assert local_infos[0][0] == expect_ip
    assert local_infos[0][1] == 2046

    llm_cluster.append_remote_ip_info((expect_ip, 2046))
    remote_infos = llm_cluster.remote_ip_infos
    assert isinstance(remote_infos, (tuple, list)) and len(remote_infos) == 1
    assert isinstance(remote_infos[0], (tuple, list)) and len(remote_infos[0]) == 2
    assert remote_infos[0][0] == expect_ip
    assert remote_infos[0][1] == 2046

    llm_cluster.append_remote_ip_info(("123456", 2046))
    remote_infos = llm_cluster.remote_ip_infos
    assert isinstance(remote_infos, (tuple, list)) and len(remote_infos) == 2
    assert isinstance(remote_infos[1], (tuple, list)) and len(remote_infos[1]) == 2
    assert remote_infos[1][0] == 123456
    assert remote_infos[1][1] == 2046


# ============================ LLMEngine ============================
def test_lite_llm_engine_llm_engine_role_type_check():
    with pytest.raises(TypeError) as raise_info:
        _ = mslite.LLMEngine("abc", 0, "manual")
    assert "role must be LLMRole, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_cluster_id_type_check():
    with pytest.raises(TypeError) as raise_info:
        _ = mslite.LLMEngine(mslite.LLMRole.Prompt, "0", "manual")
    assert "cluster_id must be int, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_batch_mode_type_check():
    with pytest.raises(TypeError) as raise_info:
        _ = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, 123)
    assert "batch_mode must be str, but got" in str(raise_info.value)
    with pytest.raises(ValueError) as raise_info:
        _ = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "123")
    assert "batch_mode should be str \"auto\" or \"manual\", but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_add_model_model_paths_type_check():
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model("123.mindir", {}, None)
    assert "model_paths must be tuple/list of str, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model([123], {}, None)
    assert "model_paths element must be str, but got" in str(raise_info.value)
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["123.mindir"], {}, None)
    assert "model_paths 123.mindir at index 0 does not exist!" in str(raise_info.value)

    with open("llm_tmp.mindir", "w") as fp:
        fp.write("test mindir")
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": "456"})
    assert "Failed to add_model" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_add_model_options_type_check():
    with open("llm_tmp.mindir", "w") as fp:
        fp.write("test mindir")

    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], 123, None)
    assert "options must be dict, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {123: "456"}, None)
    assert "options key must be str, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": 456}, None)
    assert "options value must be str, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": 456}, None)
    assert "options value must be str, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_add_model_postprocess_model_type_check():
    with open("llm_tmp.mindir", "w") as fp:
        fp.write("test mindir")
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": "456"}, 123)
    assert "postprocess_model_path must be None or str, but got" in str(raise_info.value)
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": "456"}, "123.mindir")
    assert "postprocess_model_path 123.mindir does not exist" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_init_options_type_check():
    with open("llm_tmp.mindir", "w") as fp:
        fp.write("test mindir")
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.init({"123": "456"})
    assert "At least one group of models need to be added through LLMEngine.add_model before" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], ["123", "456"])
        llm_engine.init(123)
    assert "options must be dict, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {123: "456"})
        llm_engine.init({123: "456"})
    assert "options key must be str, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.add_model(["llm_tmp.mindir"], {"123": 456})
        llm_engine.init({"123": 456})
    assert "options value must be str, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_complete_request_check():
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        cluster_id = 0
        req_id = 0
        prompt_length = 4096
        llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
        llm_engine.complete_request(llm_req)
    assert "LLMEngine is not inited or init failed" in str(raise_info.value)
    # check
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.inited_ = True
        llm_engine.complete_request("1234")
    assert "llm_req must be LLMReq, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_fetch_status_check():
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.fetch_status()
    assert "LLMEngine is not inited or init failed" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_link_clusters_check():
    cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
    cluster.append_local_ip_info(("192.168.0.1", 26000))
    cluster.append_local_ip_info(("192.168.0.2", 26000))
    cluster.append_remote_ip_info(("192.168.0.3", 26000))
    cluster.append_remote_ip_info(("192.168.0.4", 26000))
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.link_clusters([cluster])
    assert "LLMEngine is not inited or init failed" in str(raise_info.value)
    # check
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.inited_ = True
        llm_engine.link_clusters(cluster)
    assert "clusters must be list/tuple of LLMClusterInfo, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.inited_ = True
        llm_engine.link_clusters([cluster], 1.1)
    assert "timeout must be int, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_engine_unlink_clusters_check():
    cluster = mslite.LLMClusterInfo(mslite.LLMRole.Prompt, 0)
    cluster.append_local_ip_info(("192.168.0.1", 26000))
    cluster.append_local_ip_info(("192.168.0.2", 26000))
    cluster.append_remote_ip_info(("192.168.0.3", 26000))
    cluster.append_remote_ip_info(("192.168.0.4", 26000))
    with pytest.raises(RuntimeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.unlink_clusters([cluster])
    assert "LLMEngine is not inited or init failed" in str(raise_info.value)
    # check
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.inited_ = True
        llm_engine.unlink_clusters(cluster)
    assert "clusters must be list/tuple of LLMClusterInfo, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_engine = mslite.LLMEngine(mslite.LLMRole.Prompt, 0, "manual")
        llm_engine.inited_ = True
        llm_engine.unlink_clusters([cluster], 1.1)
    assert "timeout must be int, but got" in str(raise_info.value)


# ============================ LLMModel ============================
def test_lite_llm_engine_llm_model_predict_check():
    cluster_id = 0
    req_id = 0
    prompt_length = 4096
    llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
    inputs = [mslite.Tensor(np.ones((3, 224)))]

    with pytest.raises(RuntimeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.predict([llm_req], inputs)
    assert "LLMEngine is not inited or init failed" in str(raise_info.value)

    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.predict(llm_req, inputs)
    assert "lm_req must be list/tuple of LLMReq when batch_mode is \"manual\"" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.predict(["123"], inputs)
    assert "lm_req element must be LLMReq when batch_mode is \"manual\"," in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "auto")
        llm_model.inited_ = True
        llm_model.predict([llm_req], inputs)
    assert "lm_req must be LLMReq when batch_mode is \"auto\", but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.predict([llm_req], inputs[0])
    assert "inputs must be list/tuple of Tensor, but got " in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.predict([llm_req], ["1231"])
    assert "inputs element must be Tensor, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_model_pull_kv_check():
    cluster_id = 0
    req_id = 0
    prompt_length = 4096
    llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.pull_kv([llm_req])
    assert "llm_req must be LLMReq, but got" in str(raise_info.value)
    with pytest.raises(RuntimeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "auto")
        llm_model.inited_ = True
        llm_model.pull_kv(llm_req)
    assert "LLMEngine.pull_kv is only support when batch_mode is \"manual\"" in str(raise_info.value)


def test_lite_llm_engine_llm_model_merge_kv_check():
    cluster_id = 0
    req_id = 0
    prompt_length = 4096
    llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.merge_kv([llm_req], 0, 0)
    assert "llm_req must be LLMReq, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.merge_kv(llm_req, "0", 0)
    assert "batch_index must be int, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.merge_kv(llm_req, 0, "0")
    assert "batch_id must be int, but got" in str(raise_info.value)

    with pytest.raises(RuntimeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "auto")
        llm_model.inited_ = True
        llm_model.merge_kv(llm_req, 0, 0)
    assert "LLMEngine.merge_kv is only support when batch_mode is \"manual\"" in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.merge_kv(llm_req, -1, 0)
    assert "batch_index value should be in range [0, UINT32_MAX], but got" in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.merge_kv(llm_req, 0, -1)
    assert "batch_id value should be in range [0, UINT32_MAX], but got" in str(raise_info.value)


def test_lite_llm_engine_llm_model_preload_prompt_prefix_check():
    cluster_id = 0
    req_id = 0
    prompt_length = 4096
    llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
    inputs = [mslite.Tensor(np.ones((3, 224)))]
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.preload_prompt_prefix([llm_req], inputs)
    assert "llm_req must be LLMReq, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manul")
        llm_model.inited_ = True
        llm_model.preload_prompt_prefix(llm_req, inputs[0])
    assert "inputs must be list/tuple of Tensor, but got " in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manul")
        llm_model.inited_ = True
        llm_model.preload_prompt_prefix(llm_req, ["1231"])
    assert "inputs element must be Tensor, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_model_release_prompt_prefix_check():
    cluster_id = 0
    req_id = 0
    prompt_length = 4096
    llm_req = mslite.LLMReq(cluster_id, req_id, prompt_length)
    with pytest.raises(TypeError) as raise_info:
        llm_model = mslite.llm_engine.LLMModel("fake_model_obj", "manual")
        llm_model.inited_ = True
        llm_model.release_prompt_prefix([llm_req])
    assert "llm_req must be LLMReq, but got" in str(raise_info.value)


# ============================ LLMReq ============================

def test_lite_llm_engine_llm_req_parameter_type_check():
    with pytest.raises(TypeError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.decoder_cluster_id = "123"
    assert "decoder_cluster_id must be int, but got" in str(raise_info.value)
    with pytest.raises(TypeError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.sequence_length = "123"
    assert "sequence_length must be int, but got" in str(raise_info.value)


def test_lite_llm_engine_llm_req_parameter_num_range_check():
    with pytest.raises(ValueError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.decoder_cluster_id = -1
    assert "decoder_cluster_id value should be in range [0, UINT64_MAX], but got" in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.decoder_cluster_id = pow(2, 64)
    assert "decoder_cluster_id value should be in range [0, UINT64_MAX], but got" in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.sequence_length = -1
    assert "sequence_length value should be in range [0, UINT64_MAX], but got" in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        llm_req = mslite.LLMReq(0, 0, 0)
        llm_req.sequence_length = pow(2, 64)
    assert "sequence_length value should be in range [0, UINT64_MAX], but got" in str(raise_info.value)
