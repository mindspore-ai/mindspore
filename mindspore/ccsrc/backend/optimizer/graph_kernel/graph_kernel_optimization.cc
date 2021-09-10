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
#include "utils/context/graph_kernel_flags.h"
#include "backend/optimizer/graph_kernel/add_atomic_clean.h"
#include "backend/optimizer/graph_kernel/add_stitch_atomic_clean_gpu.h"
#include "backend/optimizer/graph_kernel/arithmetic_simplify.h"
#include "backend/optimizer/graph_kernel/graph_kernel_cluster.h"
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"
#include "backend/optimizer/graph_kernel/insert_pad.h"
#include "backend/optimizer/graph_kernel/graph_kernel_splitter.h"
#include "backend/optimizer/graph_kernel/graph_kernel_expander.h"
#include "backend/optimizer/graph_kernel/cast_matmul_fusion.h"
#include "backend/optimizer/graph_kernel/raise_reduction_precision.h"
#include "backend/optimizer/graph_kernel/graph_kernel_cse.h"
#include "backend/optimizer/graph_kernel/shape_ops_splitter.h"
#include "backend/optimizer/graph_kernel/value_graph_binder.h"
#include "backend/optimizer/graph_kernel/parallel_fusion.h"
#include "backend/optimizer/graph_kernel/optimize_assign.h"
#include "backend/optimizer/graph_kernel/split_umonad.h"
#include "backend/optimizer/graph_kernel/reorder_ops.h"
#include "backend/optimizer/graph_kernel/update_state_formatter.h"
#include "backend/optimizer/graph_kernel/axis_normalizer.h"
#include "backend/optimizer/graph_kernel/decrease_compute_precision.h"
#include "backend/optimizer/graph_kernel/decrease_transfer_precision.h"
#include "backend/optimizer/graph_kernel/tsa_atomic_add_to_first_tensor.h"
#include "backend/optimizer/graph_kernel/uss_atomic_add.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "backend/optimizer/graph_kernel/graph_kernel_pass_manager.h"
#include "backend/optimizer/graph_kernel/transform_op_optimizer.h"
#include "backend/optimizer/graph_kernel/rewrite_output_shape.h"

namespace mindspore {
namespace opt {
using context::OptLevel_1;
using context::OptLevel_2;
using context::OptLevel_3;
using context::OptLevel_MAX;
namespace {
inline unsigned int GetPassLevelByFlag(bool flag) { return flag ? OptLevel_1 : OptLevel_MAX; }
}  // namespace

PassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(0, "preprocess");
  // Do cse before all passes of graphkernel
  pm->AddPass(std::make_shared<CommonSubexpressionElimination>("cse1"), OptLevel_1);

  // Save the original output info
  pm->AddPass(std::make_shared<SaveOutputShape>(), OptLevel_1);

  // Change Assign(p, a, U) to Assign(Depend(p, U), a)
  pm->AddPass(std::make_shared<SplitAssign>(), OptLevel_1, is_gpu);

  // Spread the MakeTuple input of UpdateState
  pm->AddPass(std::make_shared<SpreadUpdateState>(), OptLevel_1);
  // Eliminate the common nodes that generated in SpreadUpdateState
  pm->AddPass(std::make_shared<CommonSubexpressionElimination>("cse2"), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManager>(1, "cluster");

  // Expand complex op to composite kernels
  pm->AddPass(std::make_shared<GraphKernelComplexExpander>(), OptLevel_1, is_gpu);

  // Expand complex basic kernels to composite kernels
  pm->AddPass(std::make_shared<GraphKernelExpander>(), OptLevel_1);

  // Cluster basic kernels and composite kernels
  pm->AddPass(std::make_shared<GraphKernelCluster>(), OptLevel_1);

  // Eliminate the outputs without external user
  pm->AddPass(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt1() const {
  auto pm = std::make_shared<GraphKernelPassManager>(2, "highlevelopt1");

  // Remove redundant Cast(bias, fp16) for Matmul input
  pm->AddPass(std::make_shared<CastMatmulFusion>(), OptLevel_2, is_ascend);

  // Reorder Cast and Type-insensitive node
  pm->AddPass(std::make_shared<ReorderOps>(), OptLevel_2);

  // normalize the Reduce axis
  pm->AddPass(std::make_shared<AxisNormalizer>(), OptLevel_1);

  // Replace Assign with InplaceAssign, and replace original output with overridden parameters
  pm->AddPass(std::make_shared<OptimizeAssign>(), OptLevel_2);
  pm->AddPass(std::make_shared<EliminateRedundantOutput>(), OptLevel_2);

  // Cast the input of ReduceSum from float16 to float32 for higher precision
  pm->AddPass(std::make_shared<RaiseReductionPrecision>(), OptLevel_2);

  // Insert PadAkg and UnPadAkg Ops for MatMul
  pm->AddPass(std::make_shared<InsertPadOps>(), OptLevel_1, is_gpu);

  // Universal arithmetic simplify
  pm->AddPass(std::make_shared<ArithmeticSimplify>(), OptLevel_2, is_gpu);

  // Common subexpression elimination
  pm->AddPass(std::make_shared<GraphKernelCSE>(), OptLevel_2);

  // Eliminate unnecessary transform ops
  auto level = GetPassLevelByFlag(context::GraphKernelFlags::GetInstance().enable_trans_op_optimize);
  pm->AddPass(std::make_shared<TransformOpOptimizer>(), level, is_gpu);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Split() const {
  auto pm = std::make_shared<GraphKernelPassManager>(3, "split");
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
  pm->AddPass(std::make_shared<GraphKernelCSE>(), OptLevel_1);
  pm->AddPass(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt2() const {
  auto pm = std::make_shared<GraphKernelPassManager>(4, "highlevelopt2");
  // Enable atomic add
  pm->AddPass(std::make_shared<AtomicCleanInsertter>(), OptLevel_2);

  // Enable atomic add for stitch nodes.
  auto level = GetPassLevelByFlag(context::GraphKernelFlags::GetInstance().enable_stitch_fusion);
  pm->AddPass(std::make_shared<StitchAtomicCleanInsertter>(), level, is_gpu);

  // Enable low precision
  auto level_low_precision = GetPassLevelByFlag(context::GraphKernelFlags::GetInstance().enable_low_precision);
  pm->AddPass(std::make_shared<DecreaseTransferPrecision>(), level_low_precision);
  pm->AddPass(std::make_shared<DecreaseComputePrecision>(), level_low_precision, is_ascend);

  // Enable tsa and uss
  pm->AddPass(std::make_shared<TsaAtomicAddToFirstTensor>(), OptLevel_1);
  pm->AddPass(std::make_shared<UssAtomicAdd>(), OptLevel_1);

  return pm;
}

PassManagerPtr GraphKernelOptimizer::Combine() const {
  auto pm = std::make_shared<GraphKernelPassManager>(5, "combine");
  // Enable parallel fusion for gpu device
  auto level = GetPassLevelByFlag(context::GraphKernelFlags::GetInstance().enable_parallel_fusion);
  pm->AddPass(std::make_shared<ParallelOpFusion>(kGPUDevice, ParallelConfig(7)), level, is_gpu);

  return pm;
}

PassManagerPtr GraphKernelOptimizer::PostProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(6, "postprocess");
  // Make Tuple for the inputs of UpdateState. (the reverse of SpreadUpdateState)
  pm->AddPass(std::make_shared<ShrinkUpdateState>(), OptLevel_1);

  // Recover the original output info
  pm->AddPass(std::make_shared<GetitemTuple>(), OptLevel_1);
  pm->AddPass(std::make_shared<RewriteOutputShape>(), OptLevel_1);

  // Add the new tensors to the kernel_graph
  pm->AddPass(std::make_shared<BindValueToGraph>(), OptLevel_1);
  return pm;
}

void GraphKernelOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  optimizer->AddPassManager(PreProcess());
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
