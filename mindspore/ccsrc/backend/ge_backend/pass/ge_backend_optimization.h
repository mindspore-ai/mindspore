/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_BACKEND_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_BACKEND_OPTIMIZATION_H_
#include <memory>
#include <set>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/pass_manager/pass_manager.h"
namespace mindspore {
namespace backend {
namespace ge_backend {
namespace opt {
void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph);
void UnifyMindIRPass(const FuncGraphPtr &graph);
void GEBackendOptimization(const KernelGraphPtr &kernel_graph);
void GEBackendOptimizeACL(const KernelGraphPtr &kernel_graph);
void OptimizeGEGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo);
void GEUnifyMindIR(const KernelGraphPtr &kernel_graph);
void OptimizationWithoutBackend(const KernelGraphPtr &kernel_graph);
void EliminateIllegalDataTypePass(const KernelGraphPtr &kernel_graph);
void BackendCommonOptimization(const KernelGraphPtr &kernel_graph);
void CommonUnifyMindIR(const KernelGraphPtr &kernel_graph);
}  // namespace opt
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_BACKEND_OPTIMIZATION_H_
