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
#include "backend/optimizer/graph_kernel/graph_kernel_optimization.h"

#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "utils/ms_context.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean_gpu.h"
#include "backend/optimizer/graph_kernel/add_stitch_atomic_clean_gpu.h"
#include "backend/optimizer/graph_kernel/arithmetic_simplify.h"
#include "backend/optimizer/graph_kernel/basic_ops_fusion.h"
#include "backend/optimizer/graph_kernel/depend_formater.h"
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"
#include "backend/optimizer/graph_kernel/tensor_promotion.h"
#include "backend/optimizer/graph_kernel/graph_kernel_splitter.h"
#include "backend/optimizer/graph_kernel/graph_kernel_expander.h"
#include "backend/optimizer/graph_kernel/raise_reduction_precision.h"
#include "backend/optimizer/graph_kernel/graph_kernel_cse.h"
#include "backend/optimizer/graph_kernel/shape_ops_splitter.h"
#include "backend/optimizer/graph_kernel/value_graph_binder.h"
#include "backend/optimizer/graph_kernel/parallel_fusion.h"
#include "backend/optimizer/graph_kernel/optimize_assign.h"
#include "backend/optimizer/graph_kernel/split_assign.h"
#include "backend/optimizer/graph_kernel/reorder_ops.h"
#include "backend/optimizer/graph_kernel/update_state_formatter.h"
#include "backend/optimizer/graph_kernel/axis_normalizer.h"
#include "backend/optimizer/pass/getitem_tuple.h"

namespace mindspore {
namespace opt {
PassManagerPtr GraphKernelOptimizer::PreProcess() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage1_preprocess");
  // Change Assign(p, a, U) to Assign(Depend(p, U), a)
  pm->AddPass(std::make_shared<SplitAssign>());

  // Move the Depend nodes to the bottom of graph
  pm->AddPass(std::make_shared<DependFormater>());

  // Reorder TransData-Cast to Cast-TransData,
  if (is_ascend) {
    pm->AddPass(std::make_shared<ReorderOps>());
  }

  // Spread the MakeTuple input of UpdateState
  pm->AddPass(std::make_shared<SpreadUpdateState>());
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Cluster() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage2_cluster");
  // Expand complex basic kernels to composite kernels
  pm->AddPass(std::make_shared<GraphKernelExpander>());

  // Fuse basic kernels and composite kernels
  pm->AddPass(std::make_shared<BasicOpsFusion>());

  // Eliminate the outputs without external user
  pm->AddPass(std::make_shared<EliminateRedundantOutput>());
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt1() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage3_highlevelopt1");
  // normalize the Reduce axis
  pm->AddPass(std::make_shared<AxisNormalizer>());

  // Replace Assign with InplaceAssign, and replace original output with overridden parameters
  pm->AddPass(std::make_shared<OptimizeAssign>());
  pm->AddPass(std::make_shared<EliminateRedundantOutput>());

  // Cast the input of ReduceSum from float16 to float32 for higher precision*/
  pm->AddPass(std::make_shared<RaiseReductionPrecision>());

  // Universal arithmetic simplify
  if (is_gpu) {
    pm->AddPass(std::make_shared<ArithmeticSimplify>());
  }

  // Common subexpression elimination
  pm->AddPass(std::make_shared<GraphKernelCSE>());
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Split() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage4_split");
  // Move the non-scalar tensor (in composite node) to parameter list
  pm->AddPass(std::make_shared<TensorPromotion>());

  // Make certain nodes redundant so that they are used by only one user,
  // which can avoid unnecessary input-output and get better performance.
  if (is_gpu) {
    // preprocess for ShapeOpsSplitter
    pm->AddPass(std::make_shared<ExtendOutputForUpdateState>());
    std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape, prim::kPrimExpandDims, prim::kPrimCast};
    pm->AddPass(std::make_shared<ShapeOpsSplitter>(duplicated_ops));
  }

  // Split kernel according to costmodel
  pm->AddPass(std::make_shared<GraphKernelSplitter>());

  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->AddPass(std::make_shared<GetitemTuple>());

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  if (is_gpu) {
    pm->AddPass(std::make_shared<MergeOutputForUpdateState>());
    pm->AddPass(std::make_shared<GraphKernelCSE>());
    pm->AddPass(std::make_shared<EliminateRedundantOutput>());
  }
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt2() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage5_highlevelopt2");
  // Enable atomic add
  if (is_gpu) {
    pm->AddPass(std::make_shared<AtomicCleanInsertter>());
    pm->AddPass(std::make_shared<StitchAtomicCleanInsertter>());
  } else /* if (is_ascend) */ {
    pm->AddPass(std::make_shared<CleanAddAtomic>());
  }
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Combine() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage6_combine");
  // Enable parallel fusion
  if (is_gpu) {
    // Prevent fake loop in parallel fusion
    pm->AddPass(std::make_shared<DependFormater>());
    // Do parallel fusion for gpu device
    pm->AddPass(std::make_shared<ParallelOpFusion>(kGPUDevice, ParallelConfig(7)));
  }
  return pm;
}

PassManagerPtr GraphKernelOptimizer::PostProcess() {
  auto pm = std::make_shared<PassManager>("graphkernel_stage7_postprocess");
  // Add the new tensors to the kernel_graph
  pm->AddPass(std::make_shared<BindValueToGraph>());

  // Make Tuple for the inputs of UpdateState. (the reverse of SpreadUpdateState)
  pm->AddPass(std::make_shared<ShrinkUpdateState>());
  return pm;
}

void GraphKernelOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  if (is_gpu) {
    optimizer->AddPassManager(PreProcess());
  }
  optimizer->AddPassManager(Cluster());
  optimizer->AddPassManager(HighLevelOpt1());
  optimizer->AddPassManager(Split());
  optimizer->AddPassManager(HighLevelOpt2());
  optimizer->AddPassManager(Combine());
  optimizer->AddPassManager(PostProcess());

  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }
  (void)optimizer->Optimize(kernel_graph);
}

void GraphKernelOptimize(const KernelGraphPtr &kernel_graph) { GraphKernelOptimizer().Run(kernel_graph); }
}  // namespace opt
}  // namespace mindspore
