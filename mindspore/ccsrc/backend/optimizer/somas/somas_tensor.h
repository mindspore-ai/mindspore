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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_TENSOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_TENSOR_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "backend/optimizer/somas/somas_node.h"
#include "backend/optimizer/somas/somas_solver_pre.h"
#include "backend/optimizer/somas/somas_stream.h"

namespace mindspore {
namespace somas {
class SomasNode;
class SomasStream;

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
  kOutputOnly,
  kWorkspace,
  kGetNextOutput,
  kSummaryInput,
  kRefNodeInput,
  kRefNodeOutput,
  kUnknown
};

enum LifeLongType {
  kLifeLongNone,        // life time is from tensor start to tensor end
  kLifeLongGraphAll,    // life time is  from graph start to graph end
  kLifeLongGraphStart,  // life time is  from graph start to tensor end
  kLifeLongGraphEnd     // life time is  from tensor start to graph end
};

using SomasNodePtr = std::shared_ptr<SomasNode>;
using SomasStreamPtr = std::shared_ptr<SomasStream>;

class SomasTensor {
 public:
  using SomasTensorPtr = std::shared_ptr<SomasTensor>;

  size_t aligned_size_{0};
  LifeLongType lifelong_value_;

  bool ref_overlap_;
  bool between_streams_;
  bool contiguous_;

  lifetime_t lifetime_;
  TensorType type_;

  size_t offset_{0};
  size_t num_constraints_{0};

  std::set<SomasNodePtr> destinations_;
  std::set<SomasStreamPtr> destinationStreams_;
  unordered_map<SomasStreamPtr, size_t> max_destination_id_;
  unordered_map<SomasStreamPtr, SomasNodePtr> max_destinations_;

  // Constructors/Destructors
  explicit SomasTensor(size_t id, SomasNodePtr source_node, SomasStreamPtr source_stream, size_t real_size,
                       LifeLongType lifelong_value = kLifeLongNone);
  SomasTensor(const SomasTensor &) = delete;
  SomasTensor &operator=(const SomasTensor &) = delete;
  ~SomasTensor() = default;

  // Accessors
  const size_t &GetId() { return id_; }
  SomasNodePtr GetSourceNode() const { return source_node_; }
  SomasStreamPtr GetSourceStream() const { return source_stream_; }
  const size_t &GetOriginalSize() { return original_size_; }
  const size_t &GetAlignedSize() { return aligned_size_; }
  bool IsLifelong() { return lifelong_value_ == kLifeLongGraphAll; }
  bool IsWorkspace() { return type_ == kWorkspace; }
  bool IsOutputOnly() { return type_ == kOutputOnly; }
  size_t GetOffset() { return offset_; }
  bool IsBetweenStreams() { return between_streams_; }
  bool IsSemiLifelongStart() { return lifelong_value_ == kLifeLongGraphStart; }
  bool IsSemiLifelongEnd() { return lifelong_value_ == kLifeLongGraphEnd; }
  bool IsRefOverlap() { return ref_overlap_; }

  // Computing functions
  void SetOffset() {
    if (aligned_size_ != 0) {
      offset_ = solver_tensor_desc_->offset_;
    }
  }
  SomasSolverTensorDescPtr GetSolverTensorDesc();
  void ComputeMaxDestinationId();

 private:
  const size_t id_{0};
  const SomasNodePtr source_node_;
  SomasStreamPtr const source_stream_;
  const size_t original_size_{0};

  SomasSolverTensorDescPtr solver_tensor_desc_;
};
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_TENSOR_H_
