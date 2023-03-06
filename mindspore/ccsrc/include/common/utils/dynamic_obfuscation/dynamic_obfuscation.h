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
#include <set>
#include "load_mindir/load_model.h"
#include "include/common/visible.h"
#include "include/common/utils/utils.h"
#include "ops/core_ops.h"

namespace mindspore {
enum struct ObfCase : unsigned int { NotObfNode, OneInputNoWeightNode, OneInputWithWeightNode };
class COMMON_EXPORT DynamicObfuscator {
 public:
  DynamicObfuscator(const float obf_ratio, const int branch_control_input)
      : obf_ratio_(obf_ratio), branch_control_input_(branch_control_input) {}

  ~DynamicObfuscator() = default;

  FuncGraphPtr ObfuscateMindIR(const FuncGraphPtr &func_graph);

 private:
  void SubGraphFakeBranch(const FuncGraphPtr func_graph);
  std::string ObfuscateOpType(const AnfNodePtr &node);
  ObfCase ObfuscateOpCase(const std::string obf_type);
  CNodePtr GetControlNode(const FuncGraphPtr &func_graph, const AnfNodePtr &prev_node);
  CNodePtr RandomSeedModeControl(const FuncGraphPtr func_graph);
  CNodePtr CustomOpModeControl(const FuncGraphPtr func_graph, const AnfNodePtr &prev_node);

  bool IsTarget(const std::string &cnode_name);
  void UpdateDict(const AnfNodePtr &node, const bool isParent);
  void CheckDuplicatedParent(const AnfNodePtr &node);
  CNodePtr CheckInputNodes(const CNodePtr &node);
  void AddSwitchNode(const FuncGraphPtr fg);
  FuncGraphPtr CloneSubGraph(const std::vector<CNodePtr> &node_arr, const AnfNodePtr &parent_node);
  FuncGraphPtr BuildFakeGraph(const std::vector<CNodePtr> &node_arr, const AnfNodePtr &parent_node);
  CNodePtr BuildOneInputNoWeightNode(const FuncGraphPtr &fg, const mindspore::AnfNodePtr &input_node,
                                     const mindspore::PrimitivePtr prim_node);
  CNodePtr BuildOneInputWithWeightNode(const FuncGraphPtr &fg, const AnfNodePtr &input_node, const CNodePtr &conv_node,
                                       const AnfNodePtr &weights);
  CNodePtr AddPartialBranch(const FuncGraphPtr fg, FuncGraphPtr fg_sub, const std::vector<mindspore::CNodePtr> &nodes);
  PrimitivePtr get_random_prim(const std::string &obf_type, const mindspore::CNodePtr &node);
  bool IsValidOpNum(const int &current_num, const int &compa_num) const;
  const float obf_ratio_ = 0.01;
  const int branch_control_input_;
  bool has_build_appended_input = false;
  std::vector<bool> customized_func_results_;
  std::map<std::string, AnfNodePtr> node_dict_;
  std::stack<std::string> node_names_;
  std::stack<std::string> parent_names_;
  int used_control_node_ = 0;
  int subgraph_obf_num_ = 0;
  bool switch_branch_ = true;
  const std::vector<std::string> single_input_target_op_ = {
    kReLUOpName,     kSigmoidOpName, kReLU6OpName, kSoftplusOpName, kHSigmoidOpName, kFastGeLUOpName, kHSwishOpName,
    kSoftsignOpName, kSeLUOpName,    kTanhOpName,  kSquareOpName,   kAvgPoolOpName,  kMaxPoolOpName};
  const std::vector<std::string> single_input_with_weight_target_op_ = {kConv2DOpName, kMatMulOpName};
  const std::vector<PrimitivePtr> one_input_prim_ = {
    mindspore::prim::kPrimReLU,     mindspore::prim::kPrimSigmoid,  mindspore::prim::kPrimReLU6,
    mindspore::prim::kPrimSoftplus, mindspore::prim::kPrimHSigmoid, mindspore::prim::kPrimFastGeLU,
    mindspore::prim::kPrimHSwish,   mindspore::prim::kPrimSoftsign, mindspore::prim::kPrimSeLU,
    mindspore::prim::kPrimTanh,     mindspore::prim::kPrimSquare};
};
}  // namespace mindspore
#endif  // MINDSPORE_DYNAMIC_OBFUSCATION_H
