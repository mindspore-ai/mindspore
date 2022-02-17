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
#include "common/graph_kernel/lite_adapter/graph_kernel_optimization.h"

#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "utils/context/graph_kernel_flags.h"
#include "backend/common/pass/getitem_tuple.h"
#include "common/graph_kernel/core/graph_kernel_cluster.h"
#include "common/graph_kernel/core/graph_kernel_expander.h"
#include "common/graph_kernel/core/graph_kernel_splitter.h"
#include "common/graph_kernel/core/eliminate_redundant_output.h"
#include "common/graph_kernel/core/shape_ops_splitter.h"
#include "common/graph_kernel/core/update_state_formatter.h"
#include "common/graph_kernel/lite_adapter/akg_build.h"
#include "common/graph_kernel/lite_adapter/convert_const_input_to_attr.h"
#include "common/graph_kernel/lite_adapter/graph_kernel_pass_manager.h"

namespace mindspore::graphkernel {
using opt::GetitemTuple;
using opt::GraphOptimizer;

PassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(0, "preprocess");
  pm->AddPass(std::make_shared<ConvertConstInputToAttr>(), OptLevel_1);
  return pm;
}

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

  // Split kernel according to costmodel
  pm->AddPass(std::make_shared<GraphKernelSplitter>(), OptLevel_1);

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
  optimizer->AddPassManager(PreProcess());
  optimizer->AddPassManager(Cluster());
  optimizer->AddPassManager(Split());

  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  (void)optimizer->Optimize(kernel_graph);
  auto node_list = kernel_graph->GetOrderedCnodes();
  AnfNodePtrList anf_list;
  for (auto &node : node_list) {
    if (AnfUtils::IsGraphKernel(node)) {
      anf_list.push_back(node);
    }
  }
  graphkernel::AkgKernelBuilder gk;
  if (!gk.CompileJsonsInAnfnodes(anf_list)) {
    MS_LOG(WARNING) << "Graph kernel compile fail";
  }
}

void GraphKernelOptimize(const FuncGraphPtr &kernel_graph) { GraphKernelOptimizer().Run(kernel_graph); }
}  // namespace mindspore::graphkernel
