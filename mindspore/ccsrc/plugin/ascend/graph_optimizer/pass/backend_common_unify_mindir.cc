/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/backend_common_unify_mindir.h"
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "tools/profiler/profiling.h"
#include "include/utils/parallel_context.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/concat_outputs_for_all_gather.h"
#include "backend/common/pass/split_inputs_for_reduce_scatter.h"
#include "backend/common/pass/ir_fission/cdist_fission.h"
#include "backend/common/pass/ir_fission/batch_norm_grad_infer_fission.h"
#include "backend/common/pass/ir_fission/bn_split.h"
#include "backend/common/pass/ir_fission/bn_grad_split.h"
#include "backend/common/pass/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "backend/common/pass/ir_fusion/batchnorm_to_bninfer.h"
#include "backend/common/pass/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "backend/common/pass/mindir/add_depend_for_adamw.h"
#include "plugin/ascend/graph_optimizer/pass/mindir/ascend_mindir_op_adapter.h"
#include "backend/common/pass/mindir/renorm_split.h"
#include "backend/common/pass/mindir/space_batch_nd_attr_update.h"
#include "backend/common/pass/mindir/bn_grad_unify_mindir.h"
#include "backend/common/pass/mindir/all_to_all_unify_mindir.h"
#include "backend/common/pass/mindir/all_to_all_v_unify_mindir.h"
#include "backend/common/pass/mindir/neighbor_exchange_v2_unify_mindir.h"
#include "backend/common/pass/mindir/reduce_axis_update.h"
#include "backend/common/pass/mindir/clip_by_norm_fission.h"
#include "backend/common/pass/mindir/dropout_unify_mindir.h"
#include "backend/common/pass/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "backend/common/pass/mindir/adam_weight_decay_unify_mindir.h"
#include "backend/common/pass/other/lamb_fission.h"
#include "backend/common/pass/other/adjust_print_for_ge.h"
#include "backend/common/pass/other/add_attr_to_dump.h"
#include "backend/common/pass/other/getnext_for_ge.h"
#include "backend/common/pass/ir_fusion/flash_attention_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/grouped_matmul_assignadd_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/matmul_assignadd_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_layer_norm_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion/add_rms_norm_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/rms_norm_quant_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_rms_norm_quant_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_cast_rms_norm_cast_quant_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_cast_rms_norm_cast_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/transpose_batch_matmul_transpose_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/transpose_ext_matmul_ext_transpose_fusion.h"
#include "backend/common/pass/other/avg_pool_grad_for_ge.h"
#include "backend/common/pass/ir_fusion/mc2_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/shape_reshape_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/split_concat_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_allreduce_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_matmul_split_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_swiglu_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_swiglu_fusion_v2.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/swiglu_dynamic_quant_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/swiglu_reshape_dynamic_quant_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_add_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_allreduce_add_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_allreduce_add_rmsnorm_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/qbmm_allreduce_convert_bias.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_elemwise_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_sigmoid_add_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_sigmoid_cast_add_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_elemwise_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/remove_fa_tensor_to_tuple_ops.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/moe_init_routing_dyn_quantv2_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_split_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/addext_cast_rms_norm_cast_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_addext_split_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_add_split_fusion.h"
#include "backend/common/pass/ir_fusion/batchmatmul_reducescatter_alltoall_fusion.h"
#include "backend/common/pass/ir_fusion/alltoall_allgather_batch_matmul_fusion.h"

namespace mindspore {
namespace opt {
PassManagerPtr GetBackendCommonUnifyMindIRPassManager() {
  auto pm = std::make_shared<opt::PassManager>("unify_mindir");
  MS_EXCEPTION_IF_NULL(pm);
  pm->AddPass(std::make_shared<RenormSplit>());
  pm->AddPass(std::make_shared<opt::ReduceAxisUpdate>());
  pm->AddPass(std::make_shared<opt::SpaceToBatchNDAttrUpdate>());
  pm->AddPass(std::make_shared<opt::BatchToSpaceNDAttrUpdate>());
  pm->AddPass(std::make_shared<opt::AdamWeightDecayUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::AddDependForAdamW>());

  // Since the SparseSoftmaxCrossEntropyWithLogits operator can only use AICPU and has poor execution performance,
  // it does not take effect for the time being.
  if (IsJit()) {
    pm->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    pm->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    pm->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  } else {
    // Add PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR pass first to avoid the backward loss function
    // from the python frontend matching the pattern defined in
    // PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR.
    // TODO(hbhu_bin): In mindspore, SparseSoftmaxCrossEntropyWithLogits has different outputs based on the "is_grad"
    // attribute, but it has two outputs in CANN. These pass cann be removed when convert "is_grad" attribute to input.
    pm->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    pm->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    pm->AddPass(std::make_shared<opt::PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  }

  pm->AddPass(std::make_shared<opt::DropoutExtUnifyMindIR1>());
  pm->AddPass(std::make_shared<opt::DropoutGradExtUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::DropoutUnifyMindIR1>());
  pm->AddPass(std::make_shared<opt::DropoutGradUnifyMindIR>());
  // AllToAll & AlltoAllV
  pm->AddPass(std::make_shared<opt::NeighborExchangeUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::NeighborExchangeV2UnifyMindIR>());
  pm->AddPass(std::make_shared<opt::NeighborExchangeV2GradUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::AllToAllUnifyMindIR>());
  pm->AddPass(std::make_shared<opt::AlltoAllVUnifyMindIR>());
  // batchnorm
  pm->AddPass(std::make_shared<BnSplit>());
  pm->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  pm->AddPass(std::make_shared<BnGradSplit>());
  pm->AddPass(std::make_shared<BatchNormGrad2BNInferGrad>());
  pm->AddPass(std::make_shared<BatchNormGradInferFission>());
  pm->AddPass(std::make_shared<BatchNorm2BNInfer>());

  pm->AddPass(std::make_shared<opt::AdjustPrintForGe>());
  pm->AddPass(std::make_shared<opt::GetNextForGE>());
  pm->AddPass(std::make_shared<opt::SyncBnSplit>());
  pm->AddPass(std::make_shared<opt::SyncBnGradSplit>());
  pm->AddPass(std::make_shared<opt::AvgPoolGradForGE>());
  pm->AddPass(std::make_shared<opt::AddAttrToDump>());
  pm->AddPass(std::make_shared<opt::AscendMindIROpAdapter>());
  return pm;
}

PassManagerPtr GetBackendFusionGroupPassManager() {
  auto pm = std::make_shared<PassManager>("backend_fusion");
  MS_EXCEPTION_IF_NULL(pm);
  pm->AddFusionPass(std::make_shared<opt::ClipByNormFission>());
  pm->AddFusionPass(std::make_shared<CdistFission>());
  pm->AddFusionPass(std::make_shared<CdistGradFission>());
  pm->AddFusionPass(std::make_shared<opt::BatchMatMulReduceScatterAllToAllFusion>());
  pm->AddFusionPass(std::make_shared<opt::AllToAllAllGatherBatchMatMulFusion>());
  pm->AddFusionPass(std::make_shared<opt::LambFissionGe>());
  pm->AddFusionPass(std::make_shared<opt::MatmulReduceScatterFusion>());
  pm->AddFusionPass(std::make_shared<opt::AllGatherMatmulFusion>());
  pm->AddFusionPass(std::make_shared<opt::FlashAttentionFusionV1>());
  pm->AddFusionPass(std::make_shared<opt::FlashAttentionFusionV2>());
  pm->AddFusionPass(std::make_shared<opt::QuantBatchMatmulAllReduceFusion>());
  pm->AddFusionPass(std::make_shared<opt::MatMulAllReduceFusion>());
#ifdef ENABLE_ATB
  pm->AddFusionPass(std::make_shared<opt::GroupedMatmulAssignaddFusion>(), false);
  pm->AddFusionPass(std::make_shared<opt::MatmulAssignaddFusion>(), false);
#endif

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool infer_boost = ms_context->IsEnableInferBoost();
  pm->AddFusionPass(std::make_shared<opt::MatMulAllReduceAddRmsNormFusion>(), infer_boost);

#ifdef ENABLE_INTERNAL_KERNELS
  pm->AddFusionPass(std::make_shared<opt::AddLayernormFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddLayernormV3Fusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddLayernormExtFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::QbmmAddFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::InferenceSwiGLUFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::InferenceSwiGLUFusionV2>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::InferenceMatmulSplitFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddRmsNormDynamicQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::SwiGLUDynamicQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::SwiGLUReshapeDynamicQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddRmsNormQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddCastRmsNormCastQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::RmsNormAddQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::RmsNormQuantFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddCastRmsNormCastFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::AddExtCastRmsNormCastFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::ShapeReshapeFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::SplitConcatFusion>());
  pm->AddFusionPass(std::make_shared<opt::InferenceQbmmElemwiseFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatMulSigmoidAddFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatMulSigmoidCastAddFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatmulElemFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::QbmmAllReduceConvertBias>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::QbmmAllReduceAddFusion>());
  pm->AddFusionPass(std::make_shared<opt::RemoveFATensorToTupleOps>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::TransposeBatchMatmulTranspose>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::TransposeExtMatmulExtTranspose>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MoeInitRoutingDynQuantV2Fusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatmulSplitFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatmulAddExtSplitFusion>(), infer_boost);
  pm->AddFusionPass(std::make_shared<opt::MatmulAddSplitFusion>(), infer_boost);
#endif  // ENABLE_INTERNAL_KERNELS

  pm->AddFusionPass(std::make_shared<opt::AddRmsNormFusion>());
  if (ms_context->IsKByKExecutorMode()) {
    // Do communication op fusion before InsertTensorMoveForCommunication pass.
    // So these passes are before kernel select process, no need to generate kernel build info in them.
    if (parallel::ParallelContext::GetInstance()->enable_all_reduce_fusion()) {
      MS_LOG(INFO) << "Parallel comm_fusion of AllReduce is enabled.";
      pm->AddFusionPass(std::make_shared<opt::AllReduceFusion>());
    }
    if (parallel::ParallelContext::GetInstance()->enable_all_gather_fusion()) {
      MS_LOG(INFO) << "Parallel comm_fusion of AllGather is enabled.";
      pm->AddFusionPass(std::make_shared<opt::AllGatherFusion>());
      pm->AddFusionPass(std::make_shared<opt::ConcatOutputsForAllGather>());
    }
    if (parallel::ParallelContext::GetInstance()->enable_reduce_scatter_fusion()) {
      MS_LOG(INFO) << "Parallel comm_fusion of ReduceScatter is enabled.";
      pm->AddFusionPass(std::make_shared<opt::ReduceScatterFusion>());
      pm->AddFusionPass(std::make_shared<opt::SplitInputsForReduceScatter>());
    }
  }
  return pm;
}

void AscendUnfoldInputsForSpecialNodes(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_unfold_inputs_for_special_nodes_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
    DumpIRProto(kernel_graph,
                "before_unfold_inputs_for_special_nodes_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto unfold_inputs_pm = std::make_shared<opt::PassManager>("unfold_inputs_for_special_nodes_pm");
  unfold_inputs_pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());

  optimizer->AddPassManager(unfold_inputs_pm);
  (void)optimizer->Optimize(kernel_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_unfold_inputs_for_special_nodes_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnfoldInputsForSpecialNodes",
                                  start_time, profiler::GetClockSyscnt(), 0);
}
}  // namespace opt
}  // namespace mindspore
