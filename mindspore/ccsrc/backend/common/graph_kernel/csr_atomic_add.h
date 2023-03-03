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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CSR_ATOMIC_ADD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CSR_ATOMIC_ADD_H_

#include <memory>
#include <tuple>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/graph_kernel/add_atomic_clean.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore::graphkernel {
// Insert atomic clean node for reduce sum if any csr op is found in the graph.
class CsrAtomicAdd : public AtomicCleanInserter {
 public:
  CsrAtomicAdd() : AtomicCleanInserter("csr_atomic_add_process") {}
  ~CsrAtomicAdd() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using CsrAtomicAddPtr = std::shared_ptr<CsrAtomicAdd>;
}  // namespace mindspore::graphkernel

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CSR_ATOMIC_ADD_H_
