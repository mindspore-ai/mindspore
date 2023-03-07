/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_CSE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_CSE_H_

#include <vector>
#include "backend/common/pass/common_subexpression_elimination.h"

namespace mindspore::graphkernel {
using opt::BackendCSE;
class GraphKernelCSE : public opt::Pass {
 public:
  explicit GraphKernelCSE(const std::vector<PrimitivePtr> &black_list = {})
      : Pass("graph_kernel_cse"), black_list_(black_list) {}
  ~GraphKernelCSE() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  std::vector<PrimitivePtr> black_list_;
};

class GraphKernelBackendCSE : public BackendCSE {
 public:
  explicit GraphKernelBackendCSE(const std::vector<PrimitivePtr> &black_list = {}) : black_list_(black_list) {}
  ~GraphKernelBackendCSE() override = default;
  bool CheckEqualKernelBuildInfo(const AnfNodePtr &main, const AnfNodePtr &node) const override;
  bool CheckEqualCnodeInputs(const AnfNodePtr &main, const AnfNodePtr &node) const override;

 private:
  std::vector<PrimitivePtr> black_list_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_CSE_H_
