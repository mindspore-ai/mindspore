/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "frontend/optimizer/opt.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
#define USING_HASH_NAME "USING_HASH_NAME"
// Get the operator's path where the operator has be defined
std::string GetOpPythonPath(const OperatorName &op_name);

// Init python operator Instance
ValuePtr CreatOpInstance(const OperatorAttrs &attrs, const OperatorName &op_name, const std::string &instance_name);

AnfNodePtr CreatTypeInt(int64_t value);
AnfNodePtr CreatInt64Imm(int64_t value);
AnfNodePtr CreateInt32Tensor(int64_t value);
AnfNodePtr ValuePtrToAnfNodePtr(const ValuePtr &value_ptr);
AnfNodePtr CreateTuple(const std::vector<int64_t> &tuple);
std::string HashInstanceName(const std::string &name);

class GenerateGraph {
 public:
  explicit GenerateGraph(std::unordered_map<std::string, ValuePtr> origin_attrs)
      : name_idx_(0), origin_attrs_(origin_attrs) {}
  Status Init(const CNodePtr &cnode);
  ~GenerateGraph() = default;
  AnfNodePtr virtual_input_node() { return virtual_input_node_; }
  AnfNodePtr NewOpInst(const OperatorName &op_name, const OperatorAttrs &attrs);
  AnfNodePtr NewOpInst(const OperatorName &op_name);
  AnfNodePtr PushBack(const std::vector<AnfNodePtr> &inputs);

 private:
  CNodePtr cnode_;
  FuncGraphManagerPtr manager_;
  ScopePtr scope_;
  FuncGraphPtr func_graph_;
  AnfNodePtr virtual_input_node_;
  std::string instance_name_base_;
  int64_t name_idx_;
  std::unordered_map<std::string, ValuePtr> origin_attrs_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
