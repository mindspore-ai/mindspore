/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADD_ATTR_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADD_ATTR_H_

#include "ir/func_graph.h"
#include "include/backend/common/pass_manager/pass.h"

namespace mindspore::graphkernel {
class AddAttr : public opt::Pass {
 public:
  AddAttr() : Pass("add_attr") {}
  ~AddAttr() = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const AnfNodePtr &graph_kernel_node) const;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_ADD_ATTR_H_
