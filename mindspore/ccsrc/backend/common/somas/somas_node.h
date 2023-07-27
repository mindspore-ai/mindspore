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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_NODE_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_NODE_H_

#include <memory>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "backend/common/somas/somas_tensor.h"
#include "backend/common/somas/somas_parameter.h"

namespace mindspore {
namespace somas {
enum NodeType { kCommonNode, kCommunicationNode };

class SomasNode {
 public:
  // Public attributes (mutated in code)
  std::string scope_full_name_;

  // node's dependency including data dependency and time dependency
  std::set<std::shared_ptr<SomasNode>> ancestor_nodes_;
  // data tensor
  std::vector<SomasTensorPtr> input_tensors_;
  std::vector<SomasTensorPtr> output_tensors_;
  std::vector<SomasTensorPtr> workspace_tensors_;
  std::map<size_t, SomasParameterPtr> input_parameters_map_;
  // control tensor
  std::vector<SomasTensorPtr> control_input_tensors_;
  std::vector<SomasTensorPtr> control_output_tensors_;

  // Constructors/Destructors
  SomasNode(std::string scope_full_name, size_t id, NodeType type, const size_t &stream_id)
      : scope_full_name_(std::move(scope_full_name)), id_(id), type_(type), stream_id_(stream_id) {}
  SomasNode(const SomasNode &) = delete;
  SomasNode &operator=(const SomasNode &) = delete;
  ~SomasNode() = default;

  // Accessors
  const size_t &GetId() const { return id_; }
  const size_t &GetStreamId() const { return stream_id_; }
  const NodeType &GetType() const { return type_; }

 private:
  const size_t id_{0};
  const NodeType type_;
  const size_t stream_id_;
};
using SomasNodePtr = std::shared_ptr<SomasNode>;
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_NODE_H_
