/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/pass/ge_backend_optimization.h"

#include <memory>
#include <string>
#include <set>
#include <vector>
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "tools/profiler/profiling.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "ir/graph_utils.h"
#include "backend/ge_backend/pass/scalar_ops_output_unify_mindir.h"
#include "backend/ge_backend/pass/shape_unify_mindir.h"
#include "backend/ge_backend/pass/maketuple_unify_mindir.h"
#include "backend/ge_backend/pass/inputs_unify_mindir.h"
#include "backend/ge_backend/pass/scalar_unify_mindir.h"
#include "backend/ge_backend/pass/tuple_unify_mindir.h"

#include "backend/common/pass/erase_invalid_micro_depend.h"
#include "backend/common/pass/erase_not_cut_attr.h"
#include "backend/common/pass/switch_not_cut.h"

#include "backend/ge_backend/pass/histogram_fixed_width_fusion.h"
#include "backend/ge_backend/pass/tensor_array.h"
#include "backend/ge_backend/pass/specialized_prepare.h"
#include "backend/ge_backend/pass/centralization_mindir.h"
#include "backend/ge_backend/pass/trans_depend_value_to_int32.h"
#include "backend/ge_backend/pass/add_parallel_group_id_attr.h"
#include "backend/ge_backend/pass/ge_convert_const_input_to_tensor_input.h"
#include "backend/ge_backend/pass/remove_tensor_to_scalar_or_tuple_ops.h"
#include "backend/ge_backend/pass/all_to_all_v_for_ge.h"
#include "backend/ge_backend/pass/fa_alltoallv_parallel.h"
#include "backend/ge_backend/pass/hcom/insert_load_for_allgather.h"
#include "backend/ge_backend/pass/hcom/insert_depend_for_all_gather_ge.h"
#include "backend/ge_backend/pass/convert_condition_input_to_scalar.h"
#include "backend/ge_backend/pass/convert_data_depend_to_control_depend.h"
#include "backend/ge_backend/pass/maketuple_depend_remover.h"
#include "backend/ge_backend/pass/fused_cast_add.h"
#include "backend/ge_backend/pass/hcom/add_parallel_group_for_hcom.h"
#include "backend/ge_backend/pass/expand_dims_for_batchnorm.h"
#include "backend/ge_backend/pass/dropout_gen_mask_depend.h"
#include "backend/ge_backend/pass/add_cast_for_ge.h"
#include "backend/ge_backend/pass/promote_cast_for_binary_ops.h"
#include "backend/ge_backend/pass/nan_to_num_for_ge.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "backend/ge_backend/pass/unfold_nested_output.h"
#include "backend/ge_backend/pass/unfold_maketuple.h"
#include "backend/ge_backend/pass/broadcast_for_select.h"
#include "backend/ge_backend/pass/bce_with_logits_loss_for_ge.h"
#include "backend/ge_backend/pass/matmul_allreduce_fusion.h"
#include "backend/common/pass/custom_defined_depend.h"
#include "backend/common/pass/other/hcom/insert_tensor_move_for_hccl_op_ge.h"
#include "backend/common/pass/other/resize_bilinear_add_attr.h"
#include "backend/common/pass/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "backend/common/pass/other/process_call_inline.h"
#include "backend/common/pass/ir_fission/seed_adapter.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "backend/common/pass/other/process_partial_inline.h"
#include "backend/ge_backend/pass/expander_fallback.h"
#include "backend/common/pass/other/convert_pad_v3_paddings.h"
#include "backend/ge_backend/pass/convert_embedding_dense_grad_padding.h"
#include "backend/common/pass/mindir/renorm_split.h"
#include "backend/common/pass/mindir/reduce_axis_update.h"
#include "backend/common/pass/mindir/clip_by_norm_fission.h"
#include "backend/common/pass/mindir/space_batch_nd_attr_update.h"
#include "backend/common/pass/mindir/adam_weight_decay_unify_mindir.h"
#include "backend/common/pass/mindir/add_depend_for_adamw.h"
#include "backend/common/pass/ir_fission/cdist_fission.h"
#include "backend/common/pass/ir_fusion/batchmatmul_reducescatter_alltoall_fusion.h"
#include "backend/common/pass/ir_fusion/alltoall_allgather_batch_matmul_fusion.h"
#include "backend/common/pass/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "backend/common/pass/mindir/dropout_unify_mindir.h"
#include "backend/common/pass/mindir/all_to_all_unify_mindir.h"
#include "backend/common/pass/mindir/neighbor_exchange_v2_unify_mindir.h"
#include "backend/common/pass/mindir/all_to_all_v_unify_mindir.h"
#include "backend/common/pass/ir_fission/bn_split.h"
#include "backend/common/pass/mindir/bn_grad_unify_mindir.h"
#include "backend/common/pass/ir_fission/bn_grad_split.h"
#include "backend/common/pass/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "backend/common/pass/ir_fission/batch_norm_grad_infer_fission.h"
#include "backend/common/pass/ir_fusion/batchnorm_to_bninfer.h"
#include "backend/common/pass/other/lamb_fission.h"
#include "backend/common/pass/other/adjust_print_for_ge.h"
#include "backend/common/pass/other/getnext_for_ge.h"
#include "backend/ge_backend/pass/adaptive_max_pool2d_check.h"
#include "backend/common/pass/other/avg_pool_grad_for_ge.h"
#include "backend/common/pass/ir_fusion/mc2_fusion.h"
#include "backend/common/pass/other/add_attr_to_dump.h"
#include "backend/ge_backend/pass/ascend_mindir_op_adapter_for_ge.h"
#include "backend/common/pass/ir_fusion/flash_attention_fusion.h"
#include "backend/common/pass/convert_list_to_tuple.h"
#include "backend/common/pass/eliminate_func_data_type.h"
#include "backend/common/pass/conv_transpose_to_conv_bp.h"
#include "backend/common/pass/custom_op_reg_info_to_attr.h"
#include "backend/common/pass/inplace_assign_for_custom_op.h"
#include "backend/common/pass/convert_attr_to_unify_mindir.h"
#include "backend/common/pass/convert_dynamic_broadcast_to.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "backend/common/pass/custom_op_const_input_to_attr.h"
#include "backend/common/pass/convert_const_input_to_tensor_input.h"
#include "backend/common/pass/convert_tuple_output_to_maketuple.h"
#include "backend/common/pass/convert_unused_tuple_para_to_make_tuple.h"
#include "backend/common/pass/add_input_structural_for_py_execute.h"
#include "backend/common/pass/add_attr_to_node/add_attr_to_node.h"
#include "backend/common/pass/replace_addn_fusion.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace opt {
namespace {
void DoUnifyMindIRPass(const FuncGraphPtr &graph, const std::shared_ptr<mindspore::opt::GraphOptimizer> &optimizer) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(INFO) << "Do unify mindir pass for graph " << graph->ToString();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_before_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  (void)optimizer->Optimize(graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_end_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

void MarkRefGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Mark graph is ref graph: " << kernel_graph->graph_id();
  auto manager = kernel_graph->manager();
  if (manager == nullptr || kernel_graph->has_attr(kIsRefGraph)) {
    return;
  }
  for (const auto &node : TopoSort(kernel_graph->get_return(), SuccDeeperSimple, AlwaysInclude)) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, cnode) &&
                          common::AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
    if (!(is_side_effect && cnode->fullname_with_scope().find("optimizer") != std::string::npos)) {
      continue;
    }
    for (const auto &node_pair : manager->node_users()[cnode]) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimUpdateState)) {
        kernel_graph->set_attr(kIsRefGraph, MakeValue(true));
        MS_LOG(INFO) << "graph is ref graph: " << kernel_graph->graph_id();
        return;
      }
    }
  }
}

mindspore::opt::PassManagerPtr GetBackendCommonUnifyMindIRPassManager() {
  auto unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("unify_mindir");
  MS_EXCEPTION_IF_NULL(unify_mindir_pm);
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::RenormSplit>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ReduceAxisUpdate>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::SpaceToBatchNDAttrUpdate>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BatchToSpaceNDAttrUpdate>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AdamWeightDecayUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AddDependForAdamW>());
  // Since the SparseSoftmaxCrossEntropyWithLogits operator can only use AICPU and has poor execution performance,
  // it does not take effect for the time being.
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());

  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::DropoutExtUnifyMindIR1>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::DropoutGradExtUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::DropoutUnifyMindIR1>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::DropoutGradUnifyMindIR>());
  // AllToAll & AlltoAllV
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::NeighborExchangeUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::NeighborExchangeV2UnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::NeighborExchangeV2GradUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AllToAllUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AlltoAllVUnifyMindIR>());
  // batchnorm
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BnSplit>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BatchNormGradUnifyMindIR>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BnGradSplit>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BatchNormGrad2BNInferGrad>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BatchNormGradInferFission>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::BatchNorm2BNInfer>());

  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AdjustPrintForGe>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::GetNextForGE>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::SyncBnSplit>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::SyncBnGradSplit>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AvgPoolGradForGE>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AddAttrToDump>());
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::AscendMindIROpAdapterForGe>());
  return unify_mindir_pm;
}

mindspore::opt::PassManagerPtr GetBackendFusionGroupPassManager() {
  auto pm = std::make_shared<mindspore::opt::PassManager>("backend_fusion");
  pm->AddFusionPass(std::make_shared<mindspore::opt::ClipByNormFission>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::CdistFission>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::CdistGradFission>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::BatchMatMulReduceScatterAllToAllFusion>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::AllToAllAllGatherBatchMatMulFusion>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::LambFissionGe>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::AdaptiveMaxPool2DGeCheck>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::MatmulReduceScatterFusion>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::AllGatherMatmulFusion>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::FlashAttentionFusionV1>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::FlashAttentionFusionV2>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::QuantBatchMatmulAllReduceFusion>());
  pm->AddFusionPass(std::make_shared<mindspore::opt::MatMulAllReduceFusion>());
  return pm;
}

}  // namespace

void UnifyMindIRPass(const FuncGraphPtr &func_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("unify_mindir_pm");
  unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::EraseInvalidMicroDepend>());
  if (common::AnfAlgo::IsDynamicGraph(func_graph)) {
    unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::EraseNotCutAttr>());
  }
  optimizer->AddPassManager(unify_mindir_pm);

  DoUnifyMindIRPass(func_graph, optimizer);
  const auto &sub_graphs = func_graph->manager()->func_graphs_used_total(func_graph);
  for (const auto &sub_graph : sub_graphs) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    DoUnifyMindIRPass(sub_graph, optimizer);
  }
  (void)profiler::CollectHostInfo("GE", "Unify MindIR Pass", "UnifyMindIRPass", start_time, profiler::GetClockSyscnt(),
                                  0);
}

void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_ge_dynamic_shape_unify_mindir_graph.ir";
    DumpIR(file_name, func_graph);
    DumpIRProto(func_graph, "before_ge_dynamic_shape_unify_mindir_hwopt");
  }
#endif
  auto dynamic_unify_mindir_pm = std::make_shared<mindspore::opt::PassManager>("ge_dynamic_unify_mindir_pm");
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarOpsOutputUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ShapeUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::MakeTupleUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::InputsUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::ScalarUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<mindspore::opt::TupleUnifyMindIR>());
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  optimizer->AddPassManager(dynamic_unify_mindir_pm);
  (void)optimizer->Optimize(func_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_ge_dynamic_shape_unify_mindir_graph.ir";
    DumpIR(file_name, func_graph);
  }
#endif
  (void)profiler::CollectHostInfo("GE", "GE Dynamic Shape Unify MindIR", "GEBackend_Dynamic_UnifyMindIR", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void GEBackendOptimization(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
  PROF_START(GEBackendOptimization);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto opt_ge_pm = std::make_shared<mindspore::opt::PassManager>("opt_ge_pm");
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::HistogramFixedWidthFusion>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayAddFlowCond1>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayAddFlowCond2>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::GeTensorArrayCastIndex>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TensorArrayPrepare>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::CentralizationMindIR>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::TransDependValueToInt32>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddParallelGroupIdAttr>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::GEConvertConstInputToTensorInput>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::RemoveTensorToScalarOrTupleOps>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AlltoAllVForGE>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::FaAlltoAllvParallel>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertLoadForAllGather>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForHcclOpGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::InsertDependForAllGatherGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ConvertCondInputToScalar>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ConvertDataDependToControlDepend>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::MakeTupleDependRemover>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::FusedCastAdd>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddParallelGroupForHcom>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ExpandDimsForBatchNorm>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::DropoutGenMaskDepend>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AddCastForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::PromoteCastForBinaryOp>(prim::kPrimMul));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::NanToNumForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::ResizeBilinearAddAttr>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::AscendConvertTupleInputToDynamicInput>(true, true));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::UnfoldNestedOutput>("unfold_nested_output"));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::UnfoldMaketuple>("unfold_nested_maketuple"));
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::BroadCastForSelect>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::BCEWithLogitsLossForGe>());
  opt_ge_pm->AddPass(std::make_shared<mindspore::opt::CustomDefinedDepend>(true, kernel_graph->graph_id()));

  optimizer->AddPassManager(opt_ge_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  PROF_END(GEBackendOptimization);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
}

void GEBackendOptimizeACL(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(GEBackendOptimizeACL);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto opt_acl_pm = std::make_shared<mindspore::opt::PassManager>("opt_acl_pm");
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ProcessCallInline>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::SeedAdapter>());

  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeInsertTensorMove)) {
    opt_acl_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForHcclOpGe>());
  } else {
    opt_acl_pm->AddPass(std::make_shared<mindspore::opt::InsertTensorMoveForCommunication>());
  }
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ProcessPartialInline>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ExpanderFallback>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertPadV3Paddings>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertPadV3GradPaddings>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ConvertEmbeddingDenseGradPadding>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::ResizeBilinearAddAttr>());
  opt_acl_pm->AddPass(std::make_shared<mindspore::opt::CustomDefinedDepend>(false, kernel_graph->graph_id()));
  optimizer->AddPassManager(opt_acl_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEBackendOptimizeACL);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", start_time,
                                  profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void OptimizeGEGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(OptimizeGEGraph);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize ge graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
  }
  MarkRefGraph(graph);
  GEBackendOptimizeACL(graph);
  GEBackendOptimization(graph);

  for (auto &child_graph : graph->child_graph_order()) {
    OptimizeGEGraph(child_graph.lock(), memo);
  }
  PROF_END(OptimizeGEGraph);
  MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
}

void GEUnifyMindIR(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(GEUnifyMindIR);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
    DumpIRProto(kernel_graph, "before_unify_mindir_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  optimizer->AddPassManager(GetBackendCommonUnifyMindIRPassManager());
  optimizer->AddPassManager(GetBackendFusionGroupPassManager());
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(GEUnifyMindIR);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  (void)profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_UnifyMindIR", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void OptimizationWithoutBackend(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(OptimizationWithoutBackend);
  MS_LOG(DEBUG) << "Start OptimizationWithoutBackend for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_optimization_without_backend_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  EliminateIllegalDataTypePass(kernel_graph);
  CommonUnifyMindIR(kernel_graph);
  BackendCommonOptimization(kernel_graph);
  MS_LOG(DEBUG) << "End OptimizationWithoutBackend for kernel graph id:" << kernel_graph->graph_id();
  PROF_END(OptimizationWithoutBackend);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_optimization_without_backend_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void EliminateIllegalDataTypePass(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(EliminateIllegalDataTypePass);
  MS_LOG(INFO) << "Start eliminate illegal data type for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto pm = std::make_shared<mindspore::opt::PassManager>("common_eliminate_illegal_data_type_pm");
  pm->AddPass(std::make_shared<mindspore::opt::ConvertListToTuple>("convert_list_to_tuple"));
  pm->AddPass(std::make_shared<mindspore::opt::EliminateFuncDataType>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  PROF_END(EliminateIllegalDataTypePass);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void CommonUnifyMindIR(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(CommonUnifyMindIR);
  MS_LOG(INFO) << "start common unify mindir opt graph:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_common_unify_mindir_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<mindspore::opt::GraphOptimizer>();
  auto pm = std::make_shared<mindspore::opt::PassManager>("common_unify_mindir_pm");
  pm->AddPass(std::make_shared<mindspore::opt::ConvTransposeToConvBackpropInputPass>());
  pm->AddPass(std::make_shared<mindspore::opt::CustomOpRegInfoToAttr>());
  pm->AddPass(std::make_shared<mindspore::opt::InplaceAssignForCustomOp>());
  pm->AddPass(std::make_shared<mindspore::opt::ConvertAttrToUnifyMindIR>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  PROF_END(CommonUnifyMindIR);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_unify_mindir_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void BackendCommonOptimization(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(BackendCommonOptimization);
  MS_LOG(INFO) << "Status record: start common optimization. graph id: " << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto common_pm = std::make_shared<mindspore::opt::PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<mindspore::opt::ConvertDynamicBroadcastTo>());
  common_pm->AddPass(std::make_shared<mindspore::opt::ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<mindspore::opt::CustomOpConstInputToAttr>());
  common_pm->AddPass(std::make_shared<mindspore::opt::ConvertConstInputToTensorInputForPrint>());
  common_pm->AddPass(std::make_shared<mindspore::opt::ConvertTupleOutputToMaketuple>());
  common_pm->AddPass(std::make_shared<mindspore::opt::ConvertUnusedTupleParaToMakeTuple>());
  common_pm->AddPass(std::make_shared<mindspore::opt::AddInputStructuralForPyExecute>());
  common_pm->AddPass(std::make_shared<mindspore::opt::AddAttrToNode>());
  common_pm->AddFusionPass(std::make_shared<mindspore::opt::ReplaceAddNFusion>());

  auto optimizer = std::make_shared<mindspore::opt::GraphOptimizer>();
  optimizer->AddPassManager(common_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(BackendCommonOptimization);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end common optimization. graph id: " << kernel_graph->graph_id();
}

}  // namespace opt
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
