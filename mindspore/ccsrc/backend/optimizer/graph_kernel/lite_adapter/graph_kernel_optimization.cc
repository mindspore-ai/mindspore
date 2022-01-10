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
#include "backend/optimizer/graph_kernel/lite_adapter/graph_kernel_optimization.h"

#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "utils/context/graph_kernel_flags.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "backend/optimizer/graph_kernel/core/graph_kernel_cluster.h"
#include "backend/optimizer/graph_kernel/core/graph_kernel_expander.h"
#include "backend/optimizer/graph_kernel/core/eliminate_redundant_output.h"
#include "backend/optimizer/graph_kernel/core/shape_ops_splitter.h"
#include "backend/optimizer/graph_kernel/core/update_state_formatter.h"
#include "backend/optimizer/graph_kernel/lite_adapter/graph_kernel_pass_manager.h"

namespace mindspore::graphkernel {
using opt::GetitemTuple;
using opt::GraphOptimizer;

PassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManager>(0, "cluster");
  // Expand complex basic kernels to composite kernels
  pm->AddPass(std::make_shared<GraphKernelExpander>(), OptLevel_1);

  // Cluster basic kernels and composite kernels
  pm->AddPass(std::make_shared<GraphKernelCluster>(), OptLevel_1);

  // Eliminate the outputs without external user
  pm->AddPass(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Split() const {
  auto pm = std::make_shared<GraphKernelPassManager>(1, "split");
  // Make certain nodes redundant so that they are used by only one user,
  // which can avoid unnecessary input-output and get better performance.
  // preprocess for ShapeOpsSplitter
  pm->AddPass(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1);
  std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape};
  pm->AddPass(std::make_shared<ShapeOpsSplitter>(duplicated_ops), OptLevel_1);

  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->AddPass(std::make_shared<GetitemTuple>(), OptLevel_1);

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  pm->AddPass(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1);
  pm->AddPass(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

void GraphKernelOptimizer::Run(const FuncGraphPtr &kernel_graph) {
  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  optimizer->AddPassManager(Cluster());
  optimizer->AddPassManager(Split());

  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  (void)optimizer->Optimize(kernel_graph);
}

void GraphKernelOptimize(const FuncGraphPtr &kernel_graph) { GraphKernelOptimizer().Run(kernel_graph); }
}  // namespace mindspore::graphkernel
