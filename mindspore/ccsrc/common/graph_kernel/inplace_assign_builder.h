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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_INPLACE_ASSIGN_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_INPLACE_ASSIGN_BUILDER_H_

#include <utility>
#include <vector>
#include <string>
#include "backend/common/optimizer/optimizer.h"

namespace mindspore::graphkernel {
struct InplaceAssignerInfo {
  // inplace-assigner node, which result will be written to inplace-assignee(an input of the func graph)
  CNodePtr op_node{nullptr};
  // inplace-assigner's index among all the func graph's outputs(inplace-assigner must be an output of func graph)
  size_t real_output_index{0};
  // num of inputs of inplace-assigner's func graph
  size_t real_output_num{0};
  // inplace-assignee's index among all the inputs; if inplace-assignee is a new additional input, set it to -1
  int inplace_to_origin_input{-1};
};

struct InplaceAssignUserInfo {
  AnfNodePtr inplace_assignee_addr{nullptr};
  AnfNodePtr work_node{nullptr};
  AnfNodePtr user_node{nullptr};
  size_t user_input_idx{0};
};

class InplaceAssignBuilder : public opt::Pass {
 public:
  explicit InplaceAssignBuilder(const std::string &name = "inplace_assign_builder") : Pass(name) {}
  ~InplaceAssignBuilder() override = default;

 protected:
  virtual void CorrectKernelBuildInfo(const AnfNodePtr &composite_node,
                                      const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &inplace_infos);
  virtual CNodePtr CreateCleanCompositeNode(const InplaceAssignerInfo &op_info, const FuncGraphPtr &main_graph,
                                            TypeId dst_type);
  void CreateAssignNodeAndCorrectReturn(
    const FuncGraphPtr &sub_graph,
    const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &parameters_infos) const;
  virtual void ProcessOriginCNode(
    const AnfNodePtr &composite_node,
    const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr);
  virtual void ProcessOriginCNodeUser(
    const FuncGraphPtr &main_graph, const AnfNodePtr &composite_node,
    const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr,
    const FuncGraphManagerPtr &mng) const;
  virtual void SetTargetAttrs(const CNodePtr &) {}

 private:
  std::vector<InplaceAssignUserInfo> FindOriginCNodeUsers(
    const FuncGraphPtr &main_graph, const AnfNodePtr &composite_node,
    const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr,
    const FuncGraphManagerPtr &mng) const;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_INPLACE_ASSIGN_BUILDER_H_
