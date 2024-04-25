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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_TENSOR_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_TENSOR_H

#include <memory>
#include <set>
#include <vector>
#include <string>
#include "utils/hash_map.h"
#include "backend/common/somas/somas_solver_pre.h"

namespace mindspore {
namespace somas {
// Lifetime type
struct Lifetime {
  size_t start_;
  size_t end_;

  explicit Lifetime(size_t start = 0, size_t end = 0) : start_(start), end_(end) {}
};
using lifetime_t = struct Lifetime;

// Tensor type
enum TensorType {
  kCommon,
  kWorkspace,
  kOutputOnly,
  kGraphOutput,
  kGraphInput,
  kSummaryInput,
  kUnion,
  kControl,
  kUnknown
};

enum LifeLongType {
  kLifeLongNone,        // life time is from tensor start to tensor end
  kLifeLongGraphAll,    // life time is from graph start to graph end
  kLifeLongGraphStart,  // life time is from graph start to tensor end
  kLifeLongGraphEnd     // life time is from tensor start to graph end
};

class BACKEND_EXPORT SomasTensor {
 public:
  size_t aligned_size_{0};
  LifeLongType lifelong_value_;

  bool contiguous_;
  bool is_peak_;
  size_t can_reuse_peak_mem_;
  bool is_graph_output_;

  lifetime_t lifetime_;
  TensorType type_;

  size_t offset_{0};

  std::set<size_t> destination_nodes_;
  vector<size_t> consumer_list_;

  // Constructors/Destructors
  explicit SomasTensor(size_t id, size_t source_node_id, size_t source_stream_id, size_t ori_size, size_t aligned_size,
                       LifeLongType lifelong_value = kLifeLongNone);
  SomasTensor(const SomasTensor &) = delete;
  SomasTensor &operator=(const SomasTensor &) = delete;
  ~SomasTensor() = default;

  // Accessors
  const size_t &GetId() const { return id_; }
  size_t GetSourceNodeId() const { return source_node_id_; }
  size_t GetSourceStreamId() const { return source_stream_id_; }
  const size_t &GetOriginalSize() const { return original_size_; }
  const size_t &GetAlignedSize() const { return aligned_size_; }
  bool IsLifelong() const { return lifelong_value_ == kLifeLongGraphAll; }
  bool IsOutputOnly() const { return type_ == kOutputOnly; }
  size_t GetOffset() const { return offset_; }
  bool IsSemiLifelongStart() const { return lifelong_value_ == kLifeLongGraphStart; }
  bool IsSemiLifelongEnd() const { return lifelong_value_ == kLifeLongGraphEnd; }
  string GetTypeString();
  string GetLifelongString();
  // Computing functions
  void SetOffset() {
    if (aligned_size_ != 0) {
      offset_ = solver_tensor_desc_->offset_;
    }
  }
  SomasSolverTensorDescPtr GetSolverTensorDesc();

 private:
  const size_t id_{0};
  const size_t source_node_id_;
  const size_t source_stream_id_;
  const size_t original_size_{0};

  SomasSolverTensorDescPtr solver_tensor_desc_;
};
using SomasTensorPtr = std::shared_ptr<SomasTensor>;
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_TENSOR_H
