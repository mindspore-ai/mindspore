/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "register/op_tiling_info.h"
#include "register/op_tiling.h"

namespace optiling {
using std::make_shared;
extern "C" ge::graphStatus OpParaCalculateV2(const ge::Operator &op, OpRunInfoV2 &run_info) {
  return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus OpAtomicCalculateV2(const ge::Node &node, OpRunInfoV2 &run_info) {
  return ge::GRAPH_SUCCESS;
}

namespace utils {
OpRunInfo::OpRunInfo() {}

OpRunInfo::OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key) {}

OpRunInfo::OpRunInfo(const OpRunInfo &runinfo) {}

OpRunInfo::OpRunInfo(OpRunInfo &&runinfo) {}

OpRunInfo &OpRunInfo::operator=(const OpRunInfo &runinfo) { return *this; }

OpRunInfo &OpRunInfo::operator=(OpRunInfo &&runinfo) { return *this; }

void OpRunInfo::SetBlockDim(const uint32_t &block_dim) { return; }

uint32_t OpRunInfo::GetBlockDim() const { return 0; }

void OpRunInfo::AddWorkspace(const int64_t &workspace) { return; }

size_t OpRunInfo::GetWorkspaceNum() const { return 0; }

ge::graphStatus OpRunInfo::GetWorkspace(const size_t &idx, int64_t &workspace) const { return ge::GRAPH_SUCCESS; }

void OpRunInfo::GetAllWorkspaces(std::vector<int64_t> &workspaces) const { return; }

void OpRunInfo::SetWorkspaces(const std::vector<int64_t> &workspaces) { return; }

void OpRunInfo::InternelSetTiling(const ByteBuffer &value) { return; }

void OpRunInfo::AddTilingData(const char *_value, size_t _size) { return; }

ByteBuffer &OpRunInfo::GetAllTilingData() {
  std::shared_ptr<ByteBuffer> tiling_data = std::make_shared<ByteBuffer>();
  return *tiling_data;
}

const ByteBuffer &OpRunInfo::GetAllTilingData() const {
  std::shared_ptr<ByteBuffer> tiling_data = std::make_shared<ByteBuffer>();
  return *tiling_data;
}

void OpRunInfo::SetClearAtomic(bool clear_atomic_input) { return; }

bool OpRunInfo::GetClearAtomic() const { return true; }

void OpRunInfo::SetTilingKey(const uint64_t &new_tiling_key) { return; }

uint64_t OpRunInfo::GetTilingKey() const { return 0; }
}  // namespace utils
}  // namespace optiling
