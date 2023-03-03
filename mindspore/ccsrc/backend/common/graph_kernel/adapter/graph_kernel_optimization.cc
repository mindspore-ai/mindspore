/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"

#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "utils/ms_context.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/add_atomic_clean.h"
#include "backend/common/graph_kernel/add_stitch_atomic_clean_gpu.h"
#include "backend/common/graph_kernel/core/arithmetic_simplify.h"
#include "backend/common/graph_kernel/core/graph_kernel_cluster.h"
#include "backend/common/graph_kernel/core/eliminate_redundant_output.h"
#include "backend/common/graph_kernel/insert_pad.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_splitter_with_py.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_expander_with_py.h"
#include "backend/common/graph_kernel/adapter/callback_impl.h"
#include "backend/common/graph_kernel/cast_matmul_fusion.h"
#include "backend/common/graph_kernel/raise_reduction_precision.h"
#include "backend/common/graph_kernel/graph_kernel_cse.h"
#include "backend/common/graph_kernel/core/shape_ops_splitter.h"
#include "backend/common/graph_kernel/value_graph_binder.h"
#include "backend/common/graph_kernel/parallel_fusion.h"
#include "backend/common/graph_kernel/optimize_assign.h"
#include "backend/common/graph_kernel/split_umonad.h"
#include "backend/common/graph_kernel/reorder_ops.h"
#include "backend/common/graph_kernel/core/update_state_formatter.h"
#include "backend/common/graph_kernel/axis_normalizer.h"
#include "backend/common/graph_kernel/decrease_compute_precision.h"
#include "backend/common/graph_kernel/decrease_transfer_precision.h"
#include "backend/common/graph_kernel/csr_atomic_add.h"
#include "backend/common/graph_kernel/tsa_atomic_add_to_first_tensor.h"
#include "backend/common/graph_kernel/uss_atomic_add.h"
#include "backend/common/pass/getitem_tuple.h"
#include "backend/common/graph_kernel/core/graph_kernel_pass_manager.h"
#include "backend/common/graph_kernel/core/transform_op_optimizer.h"
#include "backend/common/graph_kernel/rewrite_output_shape.h"
#include "backend/common/graph_kernel/graph_kernel_recompute.h"
#include "backend/common/graph_kernel/reduce_fake_out_mem.h"
#include "backend/common/graph_kernel/depend_elimination.h"
#include "backend/common/graph_kernel/tensor_inplace.h"
#include "backend/common/graph_kernel/floatstatus_addn_fusion.h"
#include "backend/common/graph_kernel/parallel_optimizer.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/compact_tensor_liveness.h"
#ifdef ENABLE_AKG
#include "backend/common/graph_kernel/graph_kernel_build.h"
#endif
namespace mindspore::graphkernel {
using opt::CommonSubexpressionElimination;
using opt::GetitemTuple;
using opt::GraphOptimizer;

namespace {
auto constexpr PARALLEL_OPS_LIMIT = 7;
inline unsigned int GetPassLevelByFlag(bool flag) { return flag ? OptLevel_1 : OptLevel_MAX; }
}  // namespace

PassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(0, "preprocess");
  // Do DependElimination all passes of graphkernel
  pm->Add(std::make_shared<DependElimination>(), OptLevel_1);

  // Do cse before all passes of graphkernel
  pm->Add(std::make_shared<CommonSubexpressionElimination>("cse1"), OptLevel_1);

  // Save the original output info
  pm->Add(std::make_shared<SaveOutputShape>(), OptLevel_1);

  // Change Assign(p, a, U) to Assign(Depend(p, U), a)
  pm->Add(std::make_shared<SplitAssign>(), OptLevel_1, is_gpu);

  // Spread the MakeTuple input of UpdateState
  pm->Add(std::make_shared<SpreadUpdateState>(), OptLevel_1);

  // Parallel optimizer by UpdateState reorganization
  pm->Add(std::make_shared<ParallelOptimizer>(PARALLEL_OPS_LIMIT), OptLevel_2);

  // Eliminate the common nodes that generated in SpreadUpdateState
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManager>(1, "cluster");

  // Expand FloatStatus(AddN)
  pm->Add(std::make_shared<FloatStatusAddNFusion>(), OptLevel_2, is_gpu);

  // Expand complex basic kernels to composite kernels
  pm->Add(std::make_shared<GraphKernelExpanderWithPy>(), OptLevel_1);

  // Cluster basic kernels and composite kernels
  pm->Add(std::make_shared<GraphKernelCluster>(), OptLevel_1);

  // Eliminate the outputs without external user
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt1() const {
  auto pm = std::make_shared<GraphKernelPassManager>(2, "highlevelopt1");

  // Remove redundant Cast(bias, fp16) for Matmul input
  pm->Add(std::make_shared<CastMatmulFusion>(), OptLevel_2, is_ascend);

  // Reorder Cast and Type-insensitive node
  pm->Add(std::make_shared<ReorderOps>(), OptLevel_2);

  // normalize the Reduce axis
  pm->Add(std::make_shared<AxisNormalizer>(), OptLevel_1);

  // Cast the input of ReduceSum from float16 to float32 for higher precision
  pm->Add(std::make_shared<RaiseReductionPrecision>(), OptLevel_2);

  // Insert PadAkg and UnPadAkg Ops for MatMul
  pm->Add(std::make_shared<InsertPadOps>(), OptLevel_1, is_gpu);

  // Universal arithmetic simplify
  pm->Add(std::make_shared<ArithmeticSimplify>(), OptLevel_2, is_gpu || is_cpu);

  // Common subexpression elimination
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_2);

  // Eliminate unnecessary transform ops
  pm->Add(std::make_shared<TransformOpOptimizer>(), OptLevel_2, is_gpu || is_cpu);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Split() const {
  auto pm = std::make_shared<GraphKernelPassManager>(3, "split");
  // Make certain nodes redundant so that they are used by only one user,
  // which can avoid unnecessary input-output and get better performance.
  // preprocess for ShapeOpsSplitter
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1);
  std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape};
  pm->Add(std::make_shared<ShapeOpsSplitter>(duplicated_ops), OptLevel_1);

  // Split kernel according to costmodel
  pm->Add(std::make_shared<GraphKernelSplitterWithPy>(), OptLevel_1);

  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1);
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_1);
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt2() const {
  auto pm = std::make_shared<GraphKernelPassManager>(4, "highlevelopt2");

  auto &flags = GraphKernelFlags::GetInstance();
  // Auto recompute according to local memory burst.
  auto recompute_lv = GetPassLevelByFlag(flags.recompute_increment_threshold > 0 ||
                                         flags.recompute_peak_threshold > 0 || flags.enable_csr_fusion);
  pm->Add(std::make_shared<GraphKernelRecompute>(), recompute_lv);

  // Enable atomic add
  pm->Add(std::make_shared<AtomicCleanInserter>(), OptLevel_2, is_gpu || is_ascend);

  // Enable atomic add for stitch nodes.
  auto level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_stitch_fusion);
  pm->Add(std::make_shared<StitchAtomicCleanInserter>(), level, is_gpu);

  // Enable low precision
  auto level_low_precision = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_low_precision);
  pm->Add(std::make_shared<DecreaseTransferPrecision>(), level_low_precision);
  pm->Add(std::make_shared<DecreaseComputePrecision>(), level_low_precision, is_ascend);

  // Optimize memory
  auto memory_optimize_level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_auto_tensor_inplace);
  pm->Add(std::make_shared<TensorInplace>(), memory_optimize_level);

  // Enable tsa and uss
  pm->Add(std::make_shared<TsaAtomicAddToFirstTensor>(), OptLevel_1, is_gpu);
  pm->Add(std::make_shared<UssAtomicAdd>(), OptLevel_1, is_gpu);
  pm->Add(std::make_shared<CsrAtomicAdd>(), OptLevel_1, is_gpu);

  // Replace Assign with InplaceAssign, and replace original output with overridden parameters
  pm->Add(std::make_shared<OptimizeAssign>(), OptLevel_2);
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), std::min(recompute_lv, OptLevel_2));
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), std::min(recompute_lv, OptLevel_2));
  pm->Add(std::make_shared<EliminateRedundantOutput>(), std::min(recompute_lv, OptLevel_2));

  return pm;
}

PassManagerPtr GraphKernelOptimizer::Combine() const {
  auto pm = std::make_shared<GraphKernelPassManager>(5, "combine");
  // Enable parallel fusion for gpu device
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_parallel_fusion);
  // Atomic-add GraphKernel node may be linked directly to UpdateState, it should be spread before parallel fusion!
  pm->Add(std::make_shared<SpreadUpdateState>(), level);
  pm->Add(std::make_shared<ParallelOpFusion>(target, ParallelConfig(PARALLEL_OPS_LIMIT)), level, is_gpu || is_ascend);

  // For memory efficiency, insert UpdateState for op with no cnode/param inputs to avoid early launching
  pm->Add(std::make_shared<CompactTensorLiveness>(), OptLevel_2);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Build() const {
  auto pm = std::make_shared<GraphKernelPassManager>(6, "build");
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1);
  // Reduce fake output memory.
  pm->Add(std::make_shared<ReduceFakeOutMem>(), OptLevel_1);
  // Compile graph kernel nodes, and inline nodes if compile failed.
#ifdef ENABLE_AKG
  pm->Add(std::make_shared<GraphKernelBuild>(), OptLevel_1);
#endif
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::PostProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(7, "postprocess");
  // Make Tuple for the inputs of UpdateState. (the reverse of SpreadUpdateState)
  pm->Add(std::make_shared<ShrinkUpdateState>(), OptLevel_1);

  // Recover the original output info
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);
  pm->Add(std::make_shared<RewriteOutputShape>(), OptLevel_1);

  // Add the new tensors to the kernel_graph
  pm->Add(std::make_shared<BindValueToGraph>(), OptLevel_1);
  return pm;
}

void GraphKernelOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  is_cpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);

  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  optimizer->AddPassManager(PreProcess());
  optimizer->AddPassManager(Cluster());
  optimizer->AddPassManager(HighLevelOpt1());
  optimizer->AddPassManager(Split());
  optimizer->AddPassManager(HighLevelOpt2());
  optimizer->AddPassManager(Combine());
  optimizer->AddPassManager(Build());
  optimizer->AddPassManager(PostProcess());

  auto mng = GkUtils::GetFuncGraphManager(kernel_graph);
  GkUtils::UpdateFuncGraphManager(mng, kernel_graph);
  (void)optimizer->Optimize(kernel_graph);
}

void GraphKernelOptimize(const KernelGraphPtr &kernel_graph) {
  GraphKernelOptimizer graph_kernel_optimizer;
  graph_kernel_optimizer.Run(kernel_graph);
}

bool GraphKernelSupported(const std::vector<AnfNodePtr> &nodes) {
  static std::vector<PrimitivePtr> supported_nodes;
  if (supported_nodes.empty()) {
    supported_nodes = GraphKernelExpanderWithPy::GetExpanderOps();
    auto cluster_nodes = GraphKernelCluster::GetClusterOps();
    (void)std::copy(cluster_nodes.begin(), cluster_nodes.end(), std::back_inserter(supported_nodes));
  }
  for (const auto &node : nodes) {
    if (node != nullptr && !std::any_of(supported_nodes.begin(), supported_nodes.end(),
                                        [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); })) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::graphkernel
