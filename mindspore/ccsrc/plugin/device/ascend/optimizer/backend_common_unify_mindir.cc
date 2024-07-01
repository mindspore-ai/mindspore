/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include <memory>
#include <string>

#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "backend/common/pass/dropout_gen_mask_fusion.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "plugin/device/ascend/optimizer/ir_fission/cdist_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/tensor_scatter_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/adam_weight_decay_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/batch_norm_grad_infer_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_grad_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchnorm_to_bninfer.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "plugin/device/ascend/optimizer/ir_fusion/histogram_fixed_width_fusion.h"
#include "plugin/device/ascend/optimizer/mindir/renorm_split.h"
#include "plugin/device/ascend/optimizer/mindir/optimizer_unify_output.h"
#include "plugin/device/ascend/optimizer/mindir/space_batch_nd_attr_update.h"
#include "plugin/device/ascend/optimizer/mindir/bn_grad_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/all_to_all_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/neighbor_exchange_v2_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/quant_dtype_cast_adjust.h"
#include "plugin/device/ascend/optimizer/mindir/fse_decode_adjust.h"
#include "plugin/device/ascend/optimizer/mindir/reduce_axis_update.h"
#include "plugin/device/ascend/optimizer/mindir/clip_by_norm_fission.h"
#include "plugin/device/ascend/optimizer/mindir/specialized_prepare.h"
#include "plugin/device/ascend/optimizer/mindir/tensor_array.h"
#include "plugin/device/ascend/optimizer/mindir/dropout_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_mindir_op_adapter.h"
#include "plugin/device/ascend/optimizer/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/adam_weight_decay_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/centralization_mindir.h"
#include "plugin/device/ascend/optimizer/ge/lamb_fission.h"
#include "plugin/device/ascend/optimizer/ge/adjust_print_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/getnext_for_ge.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adaptive_max_pool2d_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/flash_attention_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/add_layer_norm_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/add_rms_norm_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/rms_norm_quant_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/add_rms_norm_quant_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/add_cast_rms_norm_cast_fusion.h"
#include "plugin/device/ascend/optimizer/ge/avg_pool_grad_for_ge.h"
#include "plugin/device/ascend/optimizer/ir_fusion/mc2_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/shape_reshape_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/matmul_allreduce_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/matmul_elemwise_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/inference_matmul_split_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/inference_swiglu_fusion.h"

namespace mindspore {
namespace opt {
void GetBackendCommonUnifyMindIRPassManager(PassManagerPtr *unify_mindir_pm) {
  MS_EXCEPTION_IF_NULL(unify_mindir_pm);
  (*unify_mindir_pm)->AddPass(std::make_shared<RenormSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::ReduceAxisUpdate>());
  (*unify_mindir_pm)->AddPass(std::make_shared<HistogramFixedWidthFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::ClipByNormFission>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::TensorArrayAddFlowCond1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::TensorArrayAddFlowCond2>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::GeTensorArrayCastIndex>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::TensorArrayPrepare>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SpaceToBatchNDAttrUpdate>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::BatchToSpaceNDAttrUpdate>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AdamWeightDecayUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<CdistFission>());
  (*unify_mindir_pm)->AddPass(std::make_shared<CdistGradFission>());

  // Since the SparseSoftmaxCrossEntropyWithLogits operator can only use AICPU and has poor execution performance,
  // it does not take effect for the time being.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool graph_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
  bool is_kbk_mode = ms_context->IsKByKExecutorMode();
  if (graph_mode) {
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  } else {
    // Add PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR pass first to avoid the backward loss function
    // from the python frontend matching the pattern defined in
    // PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR.
    // TODO(hbhu_bin): In mindspore, SparseSoftmaxCrossEntropyWithLogits has different outputs based on the "is_grad"
    // attribute, but it has two outputs in CANN. These pass cann be removed when convert "is_grad" attribute to input.
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  }

  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutExtUnifyMindIR1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutGradExtUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutUnifyMindIR1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutGradUnifyMindIR>());

  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeV2UnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeV2GradUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AllToAllUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::QuantDTypeCastAdjust>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::FSEDecodeAdjust>());
  // batchnorm
  (*unify_mindir_pm)->AddPass(std::make_shared<BnSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BnGradSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BatchNorm2BNInfer>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BatchNormGrad2BNInferGrad>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BatchNormGradInferFission>());
  // just rename primitive name
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AscendMindIROpAdapter>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutGenMaskFusion>());

  (*unify_mindir_pm)->AddPass(std::make_shared<opt::LambFissionGe>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AdjustPrintForGe>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::GetNextForGE>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SyncBnSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SyncBnGradSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AdaptiveMaxPool2DGeFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AvgPoolGradForGE>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::FlashAttentionFusionV1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::FlashAttentionFusionV2>());
  if (!is_kbk_mode) {
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::MatmulReduceScatterFusion>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::AllGatherMatmulFusion>());
  }
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::CentralizationMindIR>());
#ifdef ENABLE_INTERNAL_KERNELS
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddLayernormFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::ShapeReshapeFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddRmsNormQuantFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::RmsNormQuantFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddRmsNormFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddCastRmsNormCastFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::MatMulAllReduceFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::InferenceSwiGLUFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::InferenceMatmulSplitFusion>());
#endif  // ENABLE_INTERNAL_KERNELS
}

void AscendUnfoldInputsForSpecialNodes(const KernelGraphPtr &kernel_graph) {
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnfoldInputsForSpecialNodes", 0, 0, 0);
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
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_unfold_inputs_for_special_nodes_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnfoldInputsForSpecialNodes", 0, 0, 1);
}
}  // namespace opt
}  // namespace mindspore
