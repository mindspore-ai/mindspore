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
#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_OPTIMIZATION_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_OPTIMIZATION_H_

#include <memory>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/errorcode.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/graph_kernel/converter/graph_kernel_pass_manager_lite.h"

namespace mindspore {
namespace graphkernel {
class GraphKernelOptimizer {
 public:
  explicit GraphKernelOptimizer(const std::shared_ptr<ConverterPara> &param) : converter_param_(param) {}
  ~GraphKernelOptimizer() = default;
  void Run(const FuncGraphPtr &func_graph);

 private:
  // Pre-process
  GkPassManagerPtr PreProcess() const;
  // Cluster kernels
  GkPassManagerPtr Cluster() const;
  // Optimize 1
  GkPassManagerPtr HighLevelOpt1() const;
  // Split kernels
  GkPassManagerPtr Split() const;
  // Build akg kernel
  GkPassManagerPtr BuildKernel() const;

  std::shared_ptr<ConverterPara> converter_param_;

  bool is_cpu{false};
  bool is_ascend{false};
};
}  // namespace graphkernel

lite::STATUS GraphKernelOptimize(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param);
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_CONVERTER_GRAPH_KERNEL_OPTIMIZATION_H_
