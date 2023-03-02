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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_PASS_MANAGER_LITE_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_PASS_MANAGER_LITE_H_

#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/core/graph_kernel_pass_manager.h"

namespace mindspore::graphkernel {
using opt::PassPtr;
class GraphKernelPassManagerLite : public GraphKernelPassManager {
 public:
  using GraphKernelPassManager::GraphKernelPassManager;

 protected:
  void DumpPassIR(const FuncGraphPtr &func_graph, const std::string &pass_fullname) const override;
  bool RunPass(const FuncGraphPtr &func_graph, size_t pass_id, const PassPtr &pass) const override;
};
using GkPassManagerPtr = std::shared_ptr<GraphKernelPassManagerLite>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_PASS_MANAGER_LITE_H_
