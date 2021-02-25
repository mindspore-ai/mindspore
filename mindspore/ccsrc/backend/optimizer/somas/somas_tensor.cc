/**
 * Copyright 2020 Huawei Technologies Co., Ltd

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

#include "backend/optimizer/somas/somas_tensor.h"
#include "backend/optimizer/somas/somas_node.h"
#include "backend/optimizer/somas/somas_stream.h"
#include "backend/optimizer/somas/somas.h"

namespace mindspore {
namespace somas {
SomasTensor::SomasTensor(size_t id, SomasNodePtr source_node, SomasStreamPtr source_stream, size_t real_size,
                         LifeLongType lifelong_value)
    : lifelong_value_(lifelong_value),
      type_(kUnknown),
      offset_(0),
      id_(id),
      source_node_(source_node),
      source_stream_(source_stream),
      original_size_(real_size) {
  const size_t alignment = 512;
  const size_t alignment_complement = 31;
  aligned_size_ = (real_size > 0) ? (real_size + alignment + alignment_complement) / alignment * alignment : 0;

  solver_tensor_desc_ = std::make_shared<SomasSolverTensorDesc>(id_, aligned_size_, offset_, false);

  ref_overlap_ = false;
  between_streams_ = false;
  contiguous_ = false;
  num_constraints_ = 0;
}

SomasSolverTensorDescPtr SomasTensor::GetSolverTensorDesc() {
  if (contiguous_) {
    solver_tensor_desc_->Update(id_, aligned_size_, offset_, false, num_constraints_);
  } else {
    solver_tensor_desc_->Update(id_, aligned_size_, offset_, lifelong_value_ == kLifeLongGraphAll, num_constraints_);
  }
  if (aligned_size_ == 0) {  // ignore zero-size tensors for solver
    return nullptr;
  } else {
    return solver_tensor_desc_;
  }
}

void SomasTensor::ComputeMaxDestinationId() {
  for (const auto &node : destinations_)
    if (node->GetId() > max_destination_id_[node->GetStream()]) {
      max_destination_id_[node->GetStream()] = node->GetId();
      max_destinations_[node->GetStream()] = node;
    }
}
}  // namespace somas
}  // namespace mindspore
