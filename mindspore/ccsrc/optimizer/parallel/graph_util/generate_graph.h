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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
#define MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_

#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <map>
#include <string>
#include <utility>

#include "./common.h"
#include "optimizer/opt.h"
#include "optimizer/parallel/strategy.h"
#include "optimizer/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
#define USING_HASH_NAME "USING_HASH_NAME"
// Get the operator's path where the operator has be defined
std::string GetOpPythonPath(const OperatorName& op_name);

// Init python operator Instance
ValuePtr CreatOpInstance(const OperatorAttrs& attrs, const OperatorName& op_name, const std::string& instance_name);

AnfNodePtr CreatTypeInt(int32_t value);
AnfNodePtr CreatInt32Imm(int32_t value);
AnfNodePtr CreateInt32Tensor(int32_t value);
std::string HashInstanceName(const std::string& name);

class GenerateGraph {
 public:
  GenerateGraph() : name_idx_(0) {}
  Status Init(const CNodePtr& cnode);
  ~GenerateGraph() = default;
  AnfNodePtr virtual_input_node() { return virtual_input_node_; }
  AnfNodePtr NewOpInst(const OperatorName& op_name, const OperatorAttrs& attrs);
  AnfNodePtr NewOpInst(const OperatorName& op_name);
  AnfNodePtr PushBack(const std::vector<AnfNodePtr>& inputs);

 private:
  CNodePtr cnode_;
  FuncGraphManagerPtr manager_;
  ScopePtr scope_;
  FuncGraphPtr func_graph_;
  AnfNodePtr virtual_input_node_;
  std::string instance_name_base_;
  int64_t name_idx_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_OPTIMIZER_PARALLEL_GRAPH_UTIL_GENERATE_GRAPH_H_
