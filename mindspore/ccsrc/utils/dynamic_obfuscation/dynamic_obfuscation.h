/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_DYNAMIC_OBFUSCATION_H
#define MINDSPORE_DYNAMIC_OBFUSCATION_H

#include <vector>
#include <string>
#include <map>
#include <stack>
#include "load_mindir/load_model.h"
#include "include/common/visible.h"

namespace mindspore {
class COMMON_EXPORT DynamicObfuscator {
 public:
  DynamicObfuscator(const float obf_ratio, const int obf_password, const int append_password)
      : obf_ratio_(obf_ratio), obf_password_(obf_password), append_password_(append_password) {}

  ~DynamicObfuscator() = default;

  FuncGraphPtr ObfuscateMindIR(const FuncGraphPtr &func_graph);

 private:
  void SubGraphFakeBranch(FuncGraphPtr func_graph);
  std::string ObfuscateOpType(const AnfNodePtr &node);
  CNodePtr GetControlNode(const FuncGraphPtr &func_graph, const AnfNodePtr &prev_node);
  CNodePtr PasswordModeControl(FuncGraphPtr func_graph);
  CNodePtr CustomOpModeControl(FuncGraphPtr func_graph, const AnfNodePtr &prev_node);

  bool IsTarget(std::string &cnode_name);
  void UpdateDict(const AnfNodePtr &node, const bool isParent);
  void CheckDuplicatedParent(const AnfNodePtr &node);
  CNodePtr CheckInputNodes(const CNodePtr &node);
  void AddSwitchNode(FuncGraphPtr fg);
  FuncGraphPtr CloneSubGraph(const FuncGraphPtr &fg, const std::vector<CNodePtr> &node_arr,
                             const AnfNodePtr &parent_node);
  FuncGraphPtr BuildFakeGraph(const FuncGraphPtr &fg, const std::vector<CNodePtr> &node_arr,
                              const AnfNodePtr &parent_node);
  CNodePtr BuildReluNode(const FuncGraphPtr &fg, const AnfNodePtr &input_node);
  CNodePtr BuildSigmoidNode(const FuncGraphPtr &fg, const AnfNodePtr &input_node);
  CNodePtr BuildOneInputWithWeightNode(const FuncGraphPtr &fg, const AnfNodePtr &input_node, const CNodePtr &conv_node,
                                       const AnfNodePtr &weights);
  CNodePtr AddPartialBranch(FuncGraphPtr fg, FuncGraphPtr fg_sub, const std::vector<mindspore::CNodePtr> &nodes);

  const float obf_ratio_ = 0.01;
  const int obf_password_;
  const int append_password_;
  bool has_build_appended_input = false;
  std::vector<bool> customized_func_results_;
  std::map<std::string, AnfNodePtr> node_dict_;
  std::stack<std::string> node_names_;
  std::stack<std::string> parent_names_;
  int used_control_node_ = 0;
  int subgraph_obf_num_ = 0;
  bool switch_branch_ = true;
  const std::vector<std::string> subgraph_target_op_ = {"Conv2D-op", "ReLU-op", "Sigmoid-op", "MatMul-op"};
};
}  // namespace mindspore
#endif  // MINDSPORE_DYNAMIC_OBFUSCATION_H
