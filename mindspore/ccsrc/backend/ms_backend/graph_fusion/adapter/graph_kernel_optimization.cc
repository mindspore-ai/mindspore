/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_optimization.h"

#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "primitive/array_ops.h"
#include "ir/func_graph.h"
#include "utils/ms_context.h"
#include "include/utils/callback.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/add_atomic_clean.h"
#include "backend/ms_backend/graph_fusion/add_stitch_atomic_clean_gpu.h"
#include "backend/ms_backend/graph_fusion/core/arithmetic_simplify.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/core/eliminate_redundant_output.h"
#include "backend/ms_backend/graph_fusion/insert_pad.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_splitter_with_py.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/adapter/callback_impl.h"
#include "backend/ms_backend/graph_fusion/raise_reduction_precision.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_cse.h"
#include "backend/ms_backend/graph_fusion/core/shape_ops_splitter.h"
#include "backend/common/pass/value_graph_binder.h"
#include "backend/ms_backend/graph_fusion/parallel_fusion.h"
#include "backend/ms_backend/graph_fusion/optimize_assign.h"
#include "backend/ms_backend/graph_fusion/core/split_umonad.h"
#include "backend/ms_backend/graph_fusion/reorder_ops.h"
#include "backend/ms_backend/graph_fusion/core/update_state_formatter.h"
#include "backend/ms_backend/graph_fusion/axis_normalizer.h"
#include "backend/ms_backend/graph_fusion/csr_atomic_add.h"
#include "backend/common/pass/getitem_tuple.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_pass_manager.h"
#include "backend/ms_backend/graph_fusion/core/transform_op_optimizer.h"
#include "backend/ms_backend/graph_fusion/rewrite_output_shape.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_recompute.h"
#include "backend/ms_backend/graph_fusion/reduce_fake_out_mem.h"
#include "backend/ms_backend/graph_fusion/depend_elimination.h"
#include "backend/ms_backend/graph_fusion/tensor_inplace.h"
#include "backend/ms_backend/graph_fusion/floatstatus_fusion.h"
#include "backend/ms_backend/graph_fusion/floatstatus_addn_fusion.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_utils.h"
#include "backend/ms_backend/graph_fusion/compact_tensor_liveness.h"
#include "backend/ms_backend/graph_fusion/adapter/symbol_engine_builder.h"
#include "backend/ms_backend/graph_fusion/kernel_packet/symbol_engine_extender.h"
#include "backend/ms_backend/graph_fusion/convert_call_to_prim.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/set_infershape_functor.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/convert_data_type.h"
#include "backend/ms_backend/graph_fusion/deal_with_side_effect.h"
#include "backend/ms_backend/graph_fusion/fold_updatestate.h"
#include "backend/ms_backend/graph_fusion/transpose_matmul_fusion.h"
#include "backend/ms_backend/graph_fusion/shrink_only_shape_needed.h"
#include "backend/ms_backend/graph_fusion/depend_edge_elimination.h"
#include "backend/ms_backend/graph_fusion/add_attr.h"
#ifdef ENABLE_AKG
#include "backend/ms_backend/graph_fusion/graph_kernel_build.h"
#endif
#include "backend/ms_backend/graph_fusion/adapter/split_model_ascend.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_cpu.h"
#include "backend/ms_backend/graph_fusion/adapter/split_model_gpu.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
namespace mindspore::graphkernel {
using opt::CommonSubexpressionElimination;
using opt::GetitemTuple;
using opt::GraphOptimizer;

namespace {
auto constexpr PARALLEL_OPS_LIMIT = 7;
inline unsigned int GetPassLevelByFlag(bool flag) { return flag ? OptLevel_1 : OptLevel_MAX; }
constexpr char kDisableKernelBackoff[] = "MS_DISABLE_KERNEL_BACKOFF";
}  // namespace

void GraphKernelOptimizer::Init() const {
  // register split model here to ensure that the correct split model will be invoked
  // when import mindspore and lite in the same process
  SPLIT_MODEL_REGISTER(kAscendDevice, inner::SplitModelAscend);
  SPLIT_MODEL_REGISTER(kCPUDevice, inner::SplitModelCpu);
  SPLIT_MODEL_REGISTER(kGPUDevice, inner::SplitModelGpu);
}

PassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(0, "preprocess");
  // Remove redundant TupleGetItem to enable cluster ops before and after TupleGetItem
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);

  // convert input to attr adapter for dyn-shape
  pm->Add(std::make_shared<ConvertFrontEndToGraphKernel>(), OptLevel_1);

  // Do DependElimination all passes of graphkernel
  pm->Add(std::make_shared<DependElimination>(), OptLevel_1);

  // Do cse before all passes of graphkernel
  pm->Add(std::make_shared<CommonSubexpressionElimination>("cse1"), OptLevel_1);

  // Save the original output info
  pm->Add(std::make_shared<SaveOutputShape>(), OptLevel_1, is_gpu);

  // Change Assign(p, a, U) to Assign(Depend(p, U), a)
  pm->Add(std::make_shared<SplitAssign>(), OptLevel_1, is_gpu || is_cpu || is_dvm);

  // Spread the MakeTuple input of UpdateState
  pm->Add(std::make_shared<SpreadUpdateState>(), OptLevel_1);

  // Eliminate the common nodes that generated in SpreadUpdateState
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManager>(1, "cluster");

  // Transform Transpose + Mutmul to a single Matmul with attribute trans_a/trans_b
  pm->Add(std::make_shared<TransposeMatmulFusion>(), OptLevel_2, is_ascend);

  // Convert IsFinite and its user to FloatStatus
  pm->Add(std::make_shared<FloatStatusFusion>(), OptLevel_2, is_dvm);

  // Expand FloatStatus(AddN)
  pm->Add(std::make_shared<FloatStatusAddNFusion>(), OptLevel_2, is_gpu || is_dvm);

  // Expand complex basic kernels to composite kernels
  pm->Add(std::make_shared<GraphKernelExpanderCloud>(), OptLevel_1);

  // Cluster basic kernels and composite kernels
  pm->Add(std::make_shared<StaticShapeCluster>(), OptLevel_1);

  // Add Cast for op's inputs in mix precision case
  pm->Add(std::make_shared<ConvertMixType>(), OptLevel_1);

  // Eliminate the outputs without external user
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::HighLevelOpt1() const {
  auto pm = std::make_shared<GraphKernelPassManager>(2, "highlevelopt1");

  // Reorder Cast and Type-insensitive node
  pm->Add(std::make_shared<ReorderOps>(), OptLevel_2);

  // normalize the Reduce axis
  pm->Add(std::make_shared<AxisNormalizer>(), OptLevel_1);

  // Insert PadAkg and UnPadAkg Ops for MatMul
  pm->Add(std::make_shared<InsertPadOps>(), OptLevel_1, is_gpu);

  // Universal arithmetic simplify
  pm->Add(std::make_shared<ArithmeticSimplify>(), OptLevel_2);

  // Add Cast for op's inputs if the input data type is not supported by op
  pm->Add(std::make_shared<ConvertBFloat16>(), OptLevel_1, is_dvm);

  // Cast the input of ReduceSum from float16 to float32 for higher precision
  pm->Add(std::make_shared<RaiseReductionPrecision>(), OptLevel_2);

  // Common subexpression elimination
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_2);

  // Eliminate unnecessary transform ops
  pm->Add(std::make_shared<TransformOpOptimizer>(), OptLevel_2);
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
  // Use symbol to calculate a more precise edge relation between nodes
  pm->Add(std::make_shared<SymbolEngineBuilder>(false), OptLevel_1, is_dvm);
  // Replace sub graph output(which is only used by Shape) with sub graph input
  pm->Add(std::make_shared<ShrinkOnlyShapeNeeded>(), OptLevel_1, is_dvm);
  // Split kernel according to costmodel
  pm->Add(std::make_shared<GraphKernelSplitterWithPy>(false), OptLevel_1);
  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use GetitemTuple Pass to delete them.
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1);
  pm->Add(std::make_shared<GraphKernelCSE>(), OptLevel_1);
  pm->Add(std::make_shared<DependEdgeElimination>(), OptLevel_1, is_dvm);
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
  pm->Add(std::make_shared<AtomicCleanInserter>(), OptLevel_2, is_gpu || (is_ascend && !is_dvm));

  // Enable atomic add for stitch nodes.
  auto level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_stitch_fusion);
  pm->Add(std::make_shared<StitchAtomicCleanInserter>(), level, is_gpu);

  // Optimize memory
  auto memory_optimize_level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_auto_tensor_inplace);
  pm->Add(std::make_shared<TensorInplace>(), memory_optimize_level);

  pm->Add(std::make_shared<CsrAtomicAdd>(), OptLevel_1, is_gpu);

  // Replace original output(which is input of Assign) with overridden parameters
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
  pm->Add(std::make_shared<FoldUpdateState>(), level, is_gpu || is_ascend);
  // Atomic-add GraphKernel node may be linked directly to UpdateState, it should be spread before parallel fusion!
  pm->Add(std::make_shared<SpreadUpdateState>(), level);
  pm->Add(std::make_shared<ParallelOpFusion>(target, ParallelConfig(PARALLEL_OPS_LIMIT)), level, is_gpu || is_ascend);

  // For memory efficiency, insert UpdateState for op with no cnode/param inputs to avoid early launching
  pm->Add(std::make_shared<CompactTensorLiveness>(), OptLevel_2, is_gpu);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::Build() const {
  // DVM does not need this stage
  auto pm = std::make_shared<GraphKernelPassManager>(6, "build");
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1, !is_dvm);
  // Reduce fake output memory.
  auto only_static_shape_fusion = GetPassLevelByFlag(!GraphKernelFlags::GetInstance().enable_dynamic_shape_fusion);
  pm->Add(std::make_shared<ReduceFakeOutMem>(), only_static_shape_fusion, !is_dvm);
  // Compile graph kernel nodes, and inline nodes if compile failed.
  auto enable_dyn_level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_dynamic_shape_fusion);
  pm->Add(std::make_shared<DynamicShapeCluster>(), enable_dyn_level, is_cpu || is_gpu);
  pm->Add(std::make_shared<SymbolEngineBuilder>(true), enable_dyn_level, is_cpu || is_gpu);
  pm->Add(std::make_shared<GraphKernelSplitterWithPy>(true), enable_dyn_level, is_gpu);
#ifdef ENABLE_AKG
  pm->Add(std::make_shared<GraphKernelBuild>(), OptLevel_1, !is_dvm);
#endif
  pm->Add(std::make_shared<GeneratedDependElimination>(), OptLevel_2, is_gpu || (is_ascend && !is_dvm));
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1, !is_dvm);
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1, !is_dvm);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::PostProcess() const {
  auto pm = std::make_shared<GraphKernelPassManager>(7, "postprocess");
  // Make Tuple for the inputs of UpdateState. (the reverse of SpreadUpdateState)
  pm->Add(std::make_shared<ShrinkUpdateState>(), OptLevel_1);

  // Recover the original output info
  pm->Add(std::make_shared<GetitemTuple>(), OptLevel_1);
  pm->Add(std::make_shared<RewriteOutputShape>(), OptLevel_1, is_gpu);

  auto enable_dyn_level = GetPassLevelByFlag(GraphKernelFlags::GetInstance().enable_dynamic_shape_fusion);
  // Add infershape functor for dynamic shape graph kernel
  pm->Add(std::make_shared<SetInferShapeFunctor>(), enable_dyn_level, !is_dvm);

  // Contrary to ConvertFrontEndToGraphKernel pass, adapter for dyn-shape
  pm->Add(std::make_shared<ConvertGraphKernelToFrontEnd>(), OptLevel_1);

  // Add the new tensors to the kernel_graph
  pm->Add(std::make_shared<opt::BindValueToGraph>(), OptLevel_1);

  // Update side effect attr, update kernel graph ref pair(used in device address allocation)
  pm->Add(std::make_shared<DealWithSideEffect>(), OptLevel_1, is_dvm || is_gpu);

  // add some attribute to graph kernel for further optimization
  pm->Add(std::make_shared<AddAttr>(), OptLevel_1);

  pm->Add(std::make_shared<ConvertCallToPrim>(), OptLevel_1, is_dvm);
  return pm;
}

PassManagerPtr GraphKernelOptimizer::KernelPacket() const {
  auto pm = std::make_shared<GraphKernelPassManager>(8, "kernelpacket");
  pm->Add(std::make_shared<packet::SymbolEngineExtender>(), OptLevel_0);
  pm->Add(std::make_shared<ConvertCallToPrim>(), OptLevel_0);
  return pm;
}

void GraphKernelOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  is_cpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  is_dvm = (GraphKernelFlags::GetInstance().kernel_generator == "DVM");

  auto parent_graph = kernel_graph->parent_graph().lock();
  FuncGraphManagerPtr parent_manager = nullptr;
  if (parent_graph != nullptr && parent_graph->manager() != nullptr) {
    parent_manager = parent_graph->manager();
  }

  Init();

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

  if (parent_graph != nullptr) {
    parent_graph->set_manager(parent_manager);
  }
}

void GraphKernelOptimizer::RunKernelPacket(const KernelGraphPtr &kernel_graph) {
  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  optimizer->AddPassManager(KernelPacket());
  (void)optimizer->Optimize(kernel_graph);
}

void GraphKernelOptimize(const KernelGraphPtr &kernel_graph) {
  PROF_START(GraphKernelOptimize);
  GraphKernelOptimizer graph_kernel_optimizer;
  graph_kernel_optimizer.Run(kernel_graph);
  PROF_END(GraphKernelOptimize);
}
REGISTER_COMMON_CALLBACK(GraphKernelOptimize);

void KernelPacketOptimize(const KernelGraphPtr &kernel_graph) {
  GraphKernelOptimizer graph_kernel_optimizer;
  graph_kernel_optimizer.RunKernelPacket(kernel_graph);
}
REGISTER_COMMON_CALLBACK(KernelPacketOptimize);
}  // namespace mindspore::graphkernel
