/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include <memory>
#include <string>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/ascend/ir_fission/bn_split.h"
#include "backend/optimizer/ascend/ir_fission/bn_grad_split.h"
#include "backend/optimizer/ascend/ir_fission/batch_norm_grad_split.h"
#include "backend/optimizer/ascend/ir_fission/batch_norm_bert_fission.h"
#include "backend/optimizer/ascend/ir_fission/single_batch_norm_fission.h"
#include "backend/optimizer/ascend/ir_fission/tensor_scatter_update_fission.h"
#include "backend/optimizer/ascend/ir_fission/reduce_min_fission.h"
#include "backend/optimizer/ascend/ir_fusion/fused_batch_norm_fusion.h"
#include "backend/optimizer/ascend/ir_fission/layer_norm_grad_split.h"
#include "backend/optimizer/ascend/ir_fission/unsorted_segment_sum_fission.h"
#include "backend/optimizer/pass/communication_op_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/square_sum_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/clip_by_norm_no_div_square_sum_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_update_with_lr_rule_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/clip_by_value_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/confusion_softmax_grad_rule.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_next_mv_rule.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_next_mv_with_decay_rule.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_next_right_rule.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_update_with_lr_v2.h"
#include "backend/optimizer/ascend/ir_fusion/layer_norm_beta_gamma_backprop_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/reshape_transpose_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/transpose_reshape_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/adam_apply_one_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/adam_apply_one_with_decay_rule.h"
#include "backend/optimizer/ascend/ir_fusion/parameter_and_transop_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/refresh_parameter_format.h"
#include "backend/optimizer/ascend/ir_fusion/transpose_transdata_fusion.h"
#include "backend/optimizer/ascend/ir_fission/transdata_split.h"
#include "backend/optimizer/ascend/ir_fission/topk_split.h"
#include "backend/optimizer/ascend/ir_fusion/momentum_lossscale_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/mul_add_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/mul_addn_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/matmul_biasadd_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/remove_reshape_pair.h"
#include "backend/optimizer/ascend/ir_fusion/derelu_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/batchnorm_to_bninfer.h"
#include "backend/optimizer/ascend/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "backend/optimizer/ascend/ir_fusion/confusion_mul_grad_fusion.h"
#include "backend/optimizer/ascend/ir_fusion/softmax_grad_ext_fusion.h"
#include "backend/optimizer/ascend/format_type/insert_trans_op.h"
#include "backend/optimizer/ascend/format_type/rectify_do_mask_kernel_info.h"
#include "backend/optimizer/ascend/format_type/chang_axis_of_reduce_kernel.h"
#include "backend/optimizer/ascend/format_type/split_unsupported_transdata.h"
#include "backend/optimizer/pass/getitem_tuple.h"
#include "backend/optimizer/pass/optimize_dependence.h"
#include "backend/optimizer/pass/erase_visit_attr.h"
#include "backend/optimizer/ascend/format_type/insert_cast.h"
#include "backend/optimizer/ascend/format_type/convert_unsupported_transnode_to_aicpu.h"
#include "backend/optimizer/pass/eliminate_redundant_op.h"
#include "backend/optimizer/pass/common_subexpression_elimination.h"
#include "backend/optimizer/pass/fuse_graph_kernel.h"
#include "backend/optimizer/pass/fuse_basic.h"
#include "backend/optimizer/pass/add_atomic_clean.h"
#include "backend/optimizer/ascend/format_type/merge_cast_to_op.h"
#include "backend/optimizer/ascend/format_type/check_consistency.h"
#include "backend/optimizer/ascend/buffer_fusion/ub_pattern_fusion.h"
#include "backend/optimizer/ascend/buffer_fusion/eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/multi_output_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/conv2dbackprop_eltwise_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/conv2dbackprop_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/conv_single_in_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/conv_double_in_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/matmul_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/depthwiseconv_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/bnupdate_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/bnupdate_eltwise_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/conv_bnreduce_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/reduce_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/buffer_fusion/segment_eltwise_fusion_pass.h"
#include "backend/optimizer/ascend/format_type/deal_ref_trans_and_cast.h"
#include "backend/optimizer/ascend/enhancer/insert_memcpy_async_for_hccl_op.h"
#include "backend/optimizer/ascend/enhancer/insert_memcpy_async_for_cascade.h"
#include "backend/optimizer/ascend/enhancer/insert_pad_for_nms_with_mask.h"
#include "backend/optimizer/ascend/format_type/insert_transdata_for_runop.h"
#include "backend/optimizer/ascend/enhancer/getnext_memcpy_elimination.h"
#include "backend/optimizer/ascend/ir_fission/addn_fission.h"
#include "backend/optimizer/ascend/enhancer/insert_memcpy_async_for_getnext.h"
#include "backend/optimizer/ascend/ir_fission/batch_norm_grad_infer_fission.h"
#include "backend/optimizer/ascend/ir_fission/split_fission.h"
#include "backend/optimizer/ascend/format_type/modify_ops_attrs.h"
#include "backend/optimizer/ascend/format_type/remove_no_use_reshape_op.h"
#include "backend/optimizer/ascend/ir_fusion/add_input_to_output.h"
#include "backend/optimizer/ascend/format_type/remove_internal_output.h"
#include "backend/optimizer/ascend/ir_fission/concat_fission.h"
#include "backend/optimizer/ascend/ir_fission/pack_fission.h"
#include "backend/optimizer/ascend/enhancer/concat_outputs_for_all_gather.h"
#include "utils/ms_context.h"
#include "utils/config_manager.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"

namespace mindspore {
namespace opt {
namespace {
void AddAscendIRFusionRulesPass(PassManager *ir_fusion_pm) {
  MS_EXCEPTION_IF_NULL(ir_fusion_pm);
  ir_fusion_pm->AddPass(std::make_shared<LambUpdateWithLRRuleFusion>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVWithDecayRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextMVRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<LambNextRightRule>());
  ir_fusion_pm->AddPass(std::make_shared<LambUpdateWithLrV2>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond1Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond2Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond3Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneCond4Fusion>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond1>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond2>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond3>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond4>());
  ir_fusion_pm->AddPass(std::make_shared<AdamApplyOneWithDecayRuleCond5>());
  ir_fusion_pm->AddPass(std::make_shared<ClipByNormNoDivSquareSumFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SquareSumFusion>());
  ir_fusion_pm->AddPass(std::make_shared<ClipByValueFusion>());
}

void AddAscendIRFusionPass(PassManager *ir_fusion_pm) {
  MS_EXCEPTION_IF_NULL(ir_fusion_pm);
  ir_fusion_pm->AddPass(std::make_shared<BatchNormBertFission>());
  ir_fusion_pm->AddPass(std::make_shared<SingleBatchNormFission>());
  ir_fusion_pm->AddPass(std::make_shared<BatchNorm2BNInfer>());
  ir_fusion_pm->AddPass(std::make_shared<BatchNormGrad2BNInferGrad>());
  ir_fusion_pm->AddPass(std::make_shared<BatchNormGradInferFission>());
  ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusionV2>());
  ir_fusion_pm->AddPass(std::make_shared<SoftmaxGradExtFusionV3>());
  ir_fusion_pm->AddPass(std::make_shared<ConfusionMulGradFusion>());
  ir_fusion_pm->AddPass(std::make_shared<ConfusionSoftmaxGradRule>());
  ir_fusion_pm->AddPass(std::make_shared<ReshapeTransposeFusion>());
  ir_fusion_pm->AddPass(std::make_shared<TransposeReshapeFusion>());
  ir_fusion_pm->AddPass(std::make_shared<TopKSplit>());
  ir_fusion_pm->AddPass(std::make_shared<MomentumLossscaleFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MulAddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MulAddNFusion>());
  ir_fusion_pm->AddPass(std::make_shared<MatmulBiasaddFusion>());
  ir_fusion_pm->AddPass(std::make_shared<AddnFission>());
  ir_fusion_pm->AddPass(std::make_shared<DereluFusion>());
  ir_fusion_pm->AddPass(std::make_shared<TransposeTransDataFusion>());
  ir_fusion_pm->AddPass(std::make_shared<SplitFission>());
  ir_fusion_pm->AddPass(std::make_shared<TensorScatterUpdateFission>());
  ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
  ir_fusion_pm->AddPass(std::make_shared<PackFission>());
  ir_fusion_pm->AddPass(std::make_shared<ConcatFission>());
  ir_fusion_pm->AddPass(std::make_shared<ReduceMinFission>());
  ir_fusion_pm->AddPass(std::make_shared<UnsortSegmentSumFission>());
}
}  // namespace

void RunOpAscendDataLayout(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto data_layout_pm = std::make_shared<PassManager>("pynative_transop_pm");
  data_layout_pm->AddPass(std::make_shared<ChangeAxisOfReduceKernel>());
  data_layout_pm->AddPass(std::make_shared<RectifyDoMaskKernelInfo>());
  data_layout_pm->AddPass(std::make_shared<RunOpInsertTransData>());
  data_layout_pm->AddPass(std::make_shared<GetitemTuple>());
  data_layout_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  data_layout_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  data_layout_pm->AddPass(std::make_shared<OptimizeDependence>());
  data_layout_pm->AddPass(std::make_shared<TransDataSplit>());
  data_layout_pm->AddPass(std::make_shared<EraseVisitAttr>());
  optimizer->AddPassManager(data_layout_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendGraphKernelCommonProcess(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<GraphOptimizer>();
  MS_EXCEPTION_IF_NULL(optimizer);
  auto common_process = std::make_shared<PassManager>("graph_kernel_common_process");
  MS_EXCEPTION_IF_NULL(common_process);
  common_process->AddPass(std::make_shared<ModifyOpAttrs>());
  common_process->AddPass(std::make_shared<RemoveNoUseReshapeOp>());
  optimizer->AddPassManager(common_process);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendDataLayout(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto data_layout_pm = std::make_shared<PassManager>("transop_pm");
  data_layout_pm->AddPass(std::make_shared<RectifyDoMaskKernelInfo>());
  data_layout_pm->AddPass(std::make_shared<InsertTransOp>());
  data_layout_pm->AddPass(std::make_shared<GetitemTuple>());
  data_layout_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  data_layout_pm->AddPass(std::make_shared<RemoveReshapePair>());
  data_layout_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  data_layout_pm->AddPass(std::make_shared<OptimizeDependence>());
  data_layout_pm->AddPass(std::make_shared<TransDataSplit>());
  data_layout_pm->AddPass(std::make_shared<EraseVisitAttr>());
  data_layout_pm->AddPass(std::make_shared<RemoveInternalOutputTransOp>());
  optimizer->AddPassManager(data_layout_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendMixPrecision(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto mixed_precision_pm = std::make_shared<PassManager>("cast_pm");
  mixed_precision_pm->AddPass(std::make_shared<InsertCast>());
  mixed_precision_pm->AddPass(std::make_shared<GetitemTuple>());
  mixed_precision_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  mixed_precision_pm->AddPass(std::make_shared<EliminateRedundantOp>());
  mixed_precision_pm->AddPass(std::make_shared<OptimizeDependence>());
  mixed_precision_pm->AddPass(std::make_shared<EraseVisitAttr>());
  mixed_precision_pm->AddPass(std::make_shared<DealRefTransAndCast>());
  mixed_precision_pm->AddPass(std::make_shared<GetitemTuple>());
  mixed_precision_pm->AddPass(std::make_shared<MergeCastToOp>());
  mixed_precision_pm->AddPass(std::make_shared<LayerNormBetaGammaBackpropFusion>());
  mixed_precision_pm->AddPass(std::make_shared<EraseVisitAttr>());
  mixed_precision_pm->AddPass(std::make_shared<SplitUnsupportedTransData>());
  mixed_precision_pm->AddPass(std::make_shared<ConvertUnSupportNodeToAICPU>());
  mixed_precision_pm->AddPass(std::make_shared<RemoveInternalOutputCast>());
  optimizer->AddPassManager(mixed_precision_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void AscendBackendIRFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_ir_fusion_before" + "_graph_" +
                            std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
    DumpIRProto(kernel_graph, "before_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ir_fusion_pm = std::make_shared<PassManager>("ir_fusion_pm");
  if (context_ptr->execution_mode() == kPynativeMode) {
    ir_fusion_pm->AddPass(std::make_shared<BnSplit>());
    ir_fusion_pm->AddPass(std::make_shared<BnGradSplit>());
  } else {
    ir_fusion_pm->AddPass(std::make_shared<BatchNormGradSplit>());
    ir_fusion_pm->AddPass(std::make_shared<FusedBatchNormFusion>());
    ir_fusion_pm->AddPass(std::make_shared<FusedBatchNormMixPrecisionFusion0>());
    ir_fusion_pm->AddPass(std::make_shared<FusedBatchNormMixPrecisionFusion1>());
  }
  ir_fusion_pm->AddPass(std::make_shared<LayerNormGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<InsertPadForNMSWithMask>());
  AddAscendIRFusionRulesPass(ir_fusion_pm.get());
  AddAscendIRFusionPass(ir_fusion_pm.get());

  if (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() && ConfigManager::GetInstance().iter_num() > 1) {
    ir_fusion_pm->AddPass(std::make_shared<InsertMemcpyAsyncForGetNext>());
    ir_fusion_pm->AddPass(std::make_shared<GetitemTuple>());
    ir_fusion_pm->AddPass(std::make_shared<EraseVisitAttr>());
  }
  ir_fusion_pm->AddPass(std::make_shared<InsertMemcpyAsyncForHcclOp>());
  ir_fusion_pm->AddPass(std::make_shared<AddInputToOutput>());
  optimizer->AddPassManager(ir_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "hwopt_d_ir_fusion_after" + "_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
  }
}

void RunOpAscendBackendIRFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->ir_fusion_flag()) {
    MS_LOG(INFO) << "IRFusion is not enable, skip";
    return;
  }
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_ir_fusion_before.ir";
    DumpIR(file_path, kernel_graph);
  }
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ir_fusion_pm = std::make_shared<PassManager>("ir_fusion_pm");
  ir_fusion_pm->AddPass(std::make_shared<SplitFission>());
  ir_fusion_pm->AddPass(std::make_shared<BnSplit>());
  ir_fusion_pm->AddPass(std::make_shared<LayerNormGradSplit>());
  ir_fusion_pm->AddPass(std::make_shared<TopKSplit>());
  ir_fusion_pm->AddPass(std::make_shared<AddnFission>());
  ir_fusion_pm->AddPass(std::make_shared<InsertPadForNMSWithMask>());
  ir_fusion_pm->AddPass(std::make_shared<TensorScatterUpdateFission>());

  optimizer->AddPassManager(ir_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_ir_fusion_after.ir";
    DumpIR(file_path, kernel_graph);
  }
}

void AscendBackendOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "hwopt_d_before" + "_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
  }
  // data layout optimization
  AscendDataLayout(kernel_graph);
  // mixed precision optimization
  AscendMixPrecision(kernel_graph);
  // other optimization
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto other_pm = std::make_shared<PassManager>("other_pm");
  other_pm->AddPass(std::make_shared<AllReduceFusion>());
  other_pm->AddPass(std::make_shared<AllGatherFusion>());
  other_pm->AddPass(std::make_shared<ConcatOutputsForAllGather>());
  other_pm->AddPass(std::make_shared<ReduceScatterFusion>());
  other_pm->AddPass(std::make_shared<BroadcastFusion>());
  other_pm->AddPass(std::make_shared<InsertMemcpyAsyncForCascade>());
  other_pm->AddPass(std::make_shared<ParameterTransOpFusion>());
  other_pm->AddPass(std::make_shared<RefreshParameterFormat>());
  optimizer->AddPassManager(other_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  // buffer fusion
  AscendBackendUBFusionOptimization(kernel_graph);

  // other2 optimization
  auto optimizer2 = std::make_shared<GraphOptimizer>();
  auto other2_pm = std::make_shared<PassManager>("other2_pm");
  other2_pm->AddPass(std::make_shared<GetitemTuple>());
  other2_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  if (context_ptr->enable_task_sink() && context_ptr->loop_sink_flag() && ConfigManager::GetInstance().iter_num() > 1) {
    other2_pm->AddPass(std::make_shared<GetnextMemcpyElimination>());
  }
  other2_pm->AddPass(std::make_shared<CheckConsistency>());
  optimizer2->AddPassManager(other2_pm);
  (void)optimizer2->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();

  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "hwopt_d_end" + "_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph, true);
    DumpIRProto(kernel_graph, "after_hwopt_" + std::to_string(kernel_graph->graph_id()));
    kernel_graph->DumpFuncGraph("hwopt_d_end");
  }
}

void AscendBackendGraphKernelOpt(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                 bool is_before_kernel_select) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!(context_ptr->enable_graph_kernel())) {
    return;
  }
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_graph_kernel_opt_before_graph_" +
                            std::to_string(!is_before_kernel_select) + "_" + std::to_string(kernel_graph->graph_id()) +
                            ".ir";
    DumpIR(file_path, kernel_graph);
  }

  // Fuse graph kernels with basic ops
  FuseGraphKernel(kernel_graph, is_before_kernel_select);

  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_graph_kernel_opt_end_graph_" +
                            std::to_string(!is_before_kernel_select) + "_" + std::to_string(kernel_graph->graph_id()) +
                            ".ir";
    DumpIR(file_path, kernel_graph, true);
  }
}

void AscendBackendFuseBasicOpt(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                               bool is_before_kernel_select) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!(context_ptr->enable_graph_kernel())) {
    return;
  }
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_fuse_basic_opt_before_graph_" +
                            std::to_string(!is_before_kernel_select) + "_" + std::to_string(kernel_graph->graph_id()) +
                            ".ir";
    DumpIR(file_path, kernel_graph, true);
  }

  // Fuse basic ops with basic ops
  FuseBasic(kernel_graph, is_before_kernel_select);

  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_fuse_basic_opt_end_graph_" +
                            std::to_string(!is_before_kernel_select) + "_" + std::to_string(kernel_graph->graph_id()) +
                            ".ir";
    DumpIR(file_path, kernel_graph, true);
  }
}

void AscendBackendAddAtomicClean(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!(context_ptr->enable_graph_kernel())) {
    return;
  }
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "hwopt_d_add_atomic_clean_before" + "_graph_" +
                            std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
  }

  AddAtomicClean(kernel_graph);

  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/" + "hwopt_d_end" + "_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph, true);
  }
}

void AscendBackendUBFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->ir_fusion_flag()) {
    MS_LOG(INFO) << "UBFusion is not enable, skip";
    return;
  }
  bool save_graphs = context_ptr->save_graphs_flag();
  auto save_graphs_path = context_ptr->save_graphs_path();
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/hwopt_d_ub_fusion_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
  }
  auto fusion_id_allocator = std::make_shared<FusionIdAllocator>();
  MS_EXCEPTION_IF_NULL(fusion_id_allocator);
  fusion_id_allocator->Init();
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ub_fusion_pm = std::make_shared<PassManager>("ub_fusion_pm");
  ub_fusion_pm->AddPass(std::make_shared<Conv2DBackpropEltwiseEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<Conv2DBackpropEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvBnReduceFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvSingleInFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BnupdateEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BnupdateEltwiseEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MatmulEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvDoubleInFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ReduceEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<SegmentEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MultiOutputFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<EltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<DepthwiseConvEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<UbPatternFusion>());
  optimizer->AddPassManager(ub_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  if (save_graphs) {
    std::string file_path =
      save_graphs_path + "/hwopt_d_ub_fusion_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_path, kernel_graph);
  }
}
}  // namespace opt
}  // namespace mindspore
