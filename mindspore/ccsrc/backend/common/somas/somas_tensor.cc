/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "backend/common/somas/somas_tensor.h"
#include <map>
#include <string>

namespace mindspore {
namespace somas {
std::map<somas::TensorType, std::string> tensor_type_name_map = {
  {kCommon, "Common"},         {kWorkspace, "Workspace"},
  {kOutputOnly, "OutputOnly"}, {kGraphOutput, "GraphOutput"},
  {kGraphInput, "GraphInput"}, {kSummaryInput, "SummaryInput"},
  {kUnion, "Union"},           {kControl, "Control"},
  {kUnknown, "Unknown"}};

std::map<LifeLongType, std::string> life_long_name_map = {{kLifeLongNone, "LifeLongNone"},
                                                          {kLifeLongGraphAll, "LifeLongGraphAll"},
                                                          {kLifeLongGraphStart, "LifeLongGraphStart"},
                                                          {kLifeLongGraphEnd, "LifeLongGraphEnd"}};

SomasTensor::SomasTensor(size_t id, size_t source_node_id, size_t source_stream_id, size_t ori_size,
                         size_t aligned_size, LifeLongType lifelong_value)
    : aligned_size_(aligned_size),
      lifelong_value_(lifelong_value),
      contiguous_(false),
      is_peak_(false),
      can_reuse_peak_mem_(0),
      is_graph_output_(false),
      type_(kUnknown),
      offset_(0),
      id_(id),
      source_node_id_(source_node_id),
      source_stream_id_(source_stream_id),
      original_size_(ori_size) {
  solver_tensor_desc_ = std::make_shared<SomasSolverTensorDesc>(id_, aligned_size_, offset_, false);
}

SomasSolverTensorDescPtr SomasTensor::GetSolverTensorDesc() {
  if (contiguous_) {
    solver_tensor_desc_->Update(id_, aligned_size_, offset_, can_reuse_peak_mem_, is_graph_output_, false);
  } else {
    solver_tensor_desc_->Update(id_, aligned_size_, offset_, can_reuse_peak_mem_, is_graph_output_,
                                lifelong_value_ == kLifeLongGraphAll);
  }
  if (aligned_size_ == 0) {  // ignore zero-size tensors for solver
    return nullptr;
  } else {
    return solver_tensor_desc_;
  }
}

std::string SomasTensor::GetTypeString() { return tensor_type_name_map[type_]; }

std::string SomasTensor::GetLifelongString() { return life_long_name_map[lifelong_value_]; }
}  // namespace somas
}  // namespace mindspore
