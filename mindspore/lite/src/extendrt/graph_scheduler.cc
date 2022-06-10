/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/graph_scheduler.h"
#include "extendrt/graph_compiler.h"
ExcutionPlan GraphCompiler::Schedule(const CompileResult &compile_result) {
  ExcutionPlan execplan;
  for (auto subgraph_ : compile_result.graphs_) {
    if (KernelGraphUtils::Instance()->IsControlNode(subgraph_)) {
      // generate control kernel
      CNode &node = subgraph_->nodes[0];
      KernelInfo kernelInfo;
      kernelInfo.kernel_ = CreateControlKernel(node);
      execplan.emplace_back(kernelInfo);
    } else if (KernelGraphUtils::Instance()->IsSingleNode(subgraph_)) {
      // generate kernel
      KernelInfo kernelInfo;
      CNode &node = subgraph_->nodes[0];
      kernelInfo.kernel_ = CreateKernel(node);
      execplan.emplace_back(kernelInfo);
    } else if (KernelGraphUtils::Instance()->IsDAG(subgraph_)) {
      // generate subgraph kernel
      KernelInfo kernelInfo;
      kernelInfo.kernel_ = CreateSubgraphKernel(subgraph_);
      execplan.emplace_back(kernelInfo);
    }
  }
  return execplan;
}
