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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_NODE_H_

#include "backend/optimizer/somas/somas_stream.h"
#include "backend/optimizer/somas/somas_tensor.h"
#include "backend/optimizer/somas/somas_parameter.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace mindspore {
namespace somas {
class SomasStream;
class SomasTensor;

enum NodeType { kCommonNode, kCommunicationNode };

using SomasStreamPtr = std::shared_ptr<SomasStream>;
using SomasTensorPtr = std::shared_ptr<SomasTensor>;

class SomasNode {
 public:
  using SomasNodePtr = std::shared_ptr<SomasNode>;
  // Public attributes (mutated in code)
  std::string scope_full_name_;

  // node's dependency including data dependency and time dependency
  std::set<SomasNodePtr> ancestor_nodes_;
  std::set<SomasTensorPtr> tensors_;

  std::vector<SomasTensorPtr> input_tensors_;
  std::vector<SomasTensorPtr> output_tensors_;
  std::vector<SomasTensorPtr> workspace_tensors_;
  std::map<size_t, SomasParameterPtr> input_parameters_map_;

  std::unordered_map<int64_t, size_t> anc_stream_max_order_;

  // Constructors/Destructors
  SomasNode(size_t id, NodeType type, SomasStreamPtr stream) : id_(id), stream_(stream), type_(type) {}
  SomasNode(const SomasNode &) = delete;
  SomasNode &operator=(const SomasNode &) = delete;
  ~SomasNode() = default;

  // Accessors
  const size_t &GetId() { return id_; }
  SomasStreamPtr GetStream() { return stream_; }
  const NodeType &GetType() { return type_; }

  // Computing ancestors
  void PresetAncestorStreams(const std::vector<SomasStreamPtr> stream_vector);
  void ComputeAncestorNodes();

 private:
  const size_t id_{0};
  SomasStreamPtr const stream_;
  const NodeType type_;
};
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_SOMAS_SOMAS_NODE_H_
