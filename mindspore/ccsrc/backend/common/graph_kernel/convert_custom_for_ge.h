/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CONVERT_CUSTOM_FOR_GE_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CONVERT_CUSTOM_FOR_GE_H_

#include <map>
#include <string>
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace graphkernel {
class ConvertCustomForGE : public opt::Pass {
 public:
  ConvertCustomForGE() : Pass("convert_custom_for_ge") {}
  ~ConvertCustomForGE() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  AnfNodePtr CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  void CreateInfoDir();
  void SaveNodesInfo(const AnfNodePtrList &nodes);
  std::map<AnfNodePtr, std::string> node_json_name_;
  std::string info_dir_;
};
}  // namespace graphkernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_CONVERT_CUSTOM_FOR_GE_H_
