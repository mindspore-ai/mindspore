/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_COMMON_BACKEND_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_COMMON_BACKEND_OPTIMIZATION_H_
#include <memory>
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/pass_manager/pass_manager.h"
namespace mindspore {
namespace opt {
void BackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph);
BACKEND_COMMON_EXPORT void CommonFinalOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph);
BACKEND_COMMON_EXPORT void CommonUnifyMindIR(const std::shared_ptr<session::KernelGraph> &kernel_graph);
BACKEND_COMMON_EXPORT void AddDynamicShapeAttrPass(const std::shared_ptr<session::KernelGraph> &kernel_graph);
PassManagerPtr GetCommonUnifyMindIRPassManager();
void EliminateIllegalDataTypePass(const std::shared_ptr<session::KernelGraph> &kernel_graph);
PassManagerPtr GetEliminateIllegalDataTypePassManager();
PassManagerPtr GetBackendCommonOptimizationPassManagerPtr();
BACKEND_COMMON_EXPORT void OptimizationWithoutBackend(const std::shared_ptr<session::KernelGraph> &kernel_graph);
BACKEND_COMMON_EXPORT void OptimizationForAnyTypeKernelGraph(const std::shared_ptr<session::KernelGraph> &kernel_graph);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_PASS_MANAGER_COMMON_BACKEND_OPTIMIZATION_H_
