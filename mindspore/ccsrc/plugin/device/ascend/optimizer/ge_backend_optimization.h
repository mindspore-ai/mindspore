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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_BACKEND_OPTIMIZATION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_BACKEND_OPTIMIZATION_H_
#include <memory>
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/pass_manager.h"
namespace mindspore {
namespace opt {
void GEBackendOptimization(const KernelGraphPtr &kernel_graph);
void GEBackendOptimizeACL(const KernelGraphPtr &kernel_graph);
void GEUnifyMindIR(const KernelGraphPtr &kernel_graph);
void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph);
void GEBackendOptimizeACLAfterKernelSelect(const KernelGraphPtr &kernel_graph);
PassManagerPtr GetGEUnifyMindIRPassManager();
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_GE_BACKEND_OPTIMIZATION_H_
