/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REDUCE_FAKE_OUT_MEM_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REDUCE_FAKE_OUT_MEM_H_

#include <memory>
#include <set>
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/optimizer/pass.h"
#include "backend/common/graph_kernel/add_atomic_clean.h"
#include "ir/func_graph.h"

namespace mindspore::graphkernel {
/**
 * @brief Reduce a fake output memory from origin memory size to 1.
 */
class ReduceFakeOutMem : public opt::Pass {
 public:
  ReduceFakeOutMem() : Pass("reduce_fake_output_memory") {}
  ~ReduceFakeOutMem() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void ModifyAbstract(const AnfNodePtr &composite_node, const std::set<size_t> &fake_real_indices,
                      const AnfNodePtrList &output_list) const;
};
using ReduceFakeOutMemPtr = std::shared_ptr<ReduceFakeOutMem>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REDUCE_FAKE_OUT_MEM_H_
