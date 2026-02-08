/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PYBIND_API_GRAPH_PIPELINE_PY_H_
#define MINDSPORE_CCSRC_PYBIND_API_GRAPH_PIPELINE_PY_H_

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <optional>

#include "base/base.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace distributed {
namespace cluster {
class TCPStoreClient;
using TCPStoreClientPtr = std::shared_ptr<TCPStoreClient>;
}  // namespace cluster
}  // namespace distributed
namespace pipeline {
namespace py = pybind11;

void ResetOpId();
void ResetOpIdWithOffset();
void InitHccl();
void InitHccl(std::optional<std::string> url, int64_t timeout, uint32_t world_size, uint32_t node_id,
              distributed::cluster::TCPStoreClientPtr store);
void FinalizeHccl();
uint32_t GetHcclRankId();
uint32_t GetHcclRankSize();
void BindDeviceCtx();

// init and exec dataset sub graph
bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run);
// Build and run dataset subgraph for ms backend
bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run);

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode);
py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode);
py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len, const std::string &dec_mode);
bool PyIsCipherFile(const std::string &file_path);
void SwapCache(const py::object &host_, const py::object &device_, const py::object &block_mapping_, const bool &type);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_GRAPH_PIPELINE_PY_H_
