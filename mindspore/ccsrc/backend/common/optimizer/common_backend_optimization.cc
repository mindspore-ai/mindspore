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
#include "backend/common/optimizer/common_backend_optimization.h"
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "backend/common/pass/convert_list_to_tuple.h"
#include "backend/common/pass/eliminate_func_data_type.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "backend/common/pass/add_input_structural_for_py_execute.h"
#include "backend/common/pass/custom_op_const_input_to_attr.h"
#include "backend/common/pass/custom_op_reg_info_to_attr.h"
#include "backend/common/pass/convert_tuple_output_to_maketuple.h"
#include "backend/common/pass/convert_const_input_to_tensor_input.h"
#include "backend/common/pass/convert_tuple_input_to_dynamic_input.h"
#include "backend/common/pass/convert_const_scalar_to_tensor.h"
#include "backend/common/pass/convert_attr_to_unify_mindir.h"
#include "backend/common/pass/optimize_updatestate.h"
#include "backend/common/pass/conv_transpose_to_conv_bp.h"
#include "backend/common/pass/reduce_optimizer.h"
#include "backend/common/pass/add_dynamic_shape_attr.h"
#include "backend/common/pass/add_akg_kernel_attrs.h"
#include "backend/common/pass/inplace_assign_for_custom_op.h"
#include "backend/common/pass/flatten_concat_fission.h"
#include "backend/common/pass/add_dropout_attrs.h"
#include "backend/common/optimizer/dynamic_shape/convert_custom_op.h"
#include "backend/common/optimizer/dynamic_shape/link_custom_op.h"
#include "backend/common/pass/convert_unused_tuple_para_to_make_tuple.h"
#include "backend/common/pass/convert_dynamic_broadcast_to.h"
#include "backend/common/pass/broadcast_to_fusion.h"
#include "backend/common/pass/accumulate_n_v2_fusion.h"
#include "backend/common/pass/addn_fusion.h"
#include "backend/common/pass/argmax_min_with_value_fusion.h"
#include "backend/common/pass/batch_matmul_attr_fusion.h"
#include "backend/common/pass/concat_offset_v1_fusion.h"
#include "backend/common/pass/dynamic_rnn_fusion.h"
#include "backend/common/pass/gather_fusion.h"
#include "backend/common/pass/im2col_fusion.h"
#include "backend/common/pass/iou_fusion.h"
#include "backend/common/pass/log_fusion.h"
#include "backend/common/pass/max_pool_with_argmax_v2_fusion.h"
#include "backend/common/pass/nan_to_num_fusion.h"
#include "backend/common/pass/parallel_concat_fusion.h"
#include "backend/common/pass/ragged_tensor_to_sparse_fusion.h"
#include "backend/common/pass/resize_v2_fusion.h"
#include "backend/common/pass/sparse_concat_fusion.h"
#include "backend/common/pass/sparse_cross_fusion.h"
#include "backend/common/pass/sparse_tensor_dense_mat_mul_fusion.h"
#include "backend/common/pass/split_fusion.h"
#include "backend/common/pass/standard_normal_fusion.h"
#include "backend/common/pass/conv3d_backprop_input_padlist_fusion.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#ifdef ENABLE_DUMP_IR
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/dump_utils.h"
#include "include/backend/debug/debugger/proto_exporter.h"
#include "backend/common/session/session_basic.h"
#endif

namespace mindspore {
namespace opt {
PassManagerPtr GetBackendCommonOptimizationPassManagerPtr(const FuncGraphPtr &graph) {
  auto common_pm = std::make_shared<PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  common_pm->AddPass(std::make_shared<ConvertDynamicBroadcastTo>());
  common_pm->AddPass(std::make_shared<ReduceOptimizer>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<CustomOpConstInputToAttr>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<ConvertTupleOutputToMaketuple>());
  common_pm->AddPass(std::make_shared<ConvertUnusedTupleParaToMakeTuple>());
  common_pm->AddPass(std::make_shared<ConvertConstScalarToTensor>());
  if (graph->has_flag(kAttrMutableKernel) || graph->has_flag(kFlagEnableRunGraphBySingleOp)) {
    common_pm->AddPass(std::make_shared<ConvertTupleInputToDynamicInput>());
  }
  common_pm->AddPass(std::make_shared<FlattenConcatFission>());
  common_pm->AddPass(std::make_shared<AddDropoutAttrs>());
  common_pm->AddPass(std::make_shared<AddInputStructuralForPyExecute>());
  common_pm->AddPass(std::make_shared<BroadcastToFusion>());
  common_pm->AddPass(std::make_shared<AccumulateNV2Fusion>());
  common_pm->AddPass(std::make_shared<AddNFusion>());
  common_pm->AddPass(std::make_shared<ArgMaxWithValueFusion>());
  common_pm->AddPass(std::make_shared<ArgMinWithValueFusion>());
  common_pm->AddPass(std::make_shared<BatchMatMulAttrFusion>());
  common_pm->AddPass(std::make_shared<ConcatOffsetV1Fusion>());
  common_pm->AddPass(std::make_shared<DynamicRNNFusion>());
  common_pm->AddPass(std::make_shared<GatherFusion>());
  common_pm->AddPass(std::make_shared<Im2ColFusion>());
  common_pm->AddPass(std::make_shared<IOUFusion>());
  common_pm->AddPass(std::make_shared<LogFusion>());
  common_pm->AddPass(std::make_shared<MaxPoolWithArgmaxV2Fusion>());
  common_pm->AddPass(std::make_shared<NanToNumFusion>());
  common_pm->AddPass(std::make_shared<ParallelConcatFusion>());
  common_pm->AddPass(std::make_shared<RaggedTensorToSparseFusion>());
  common_pm->AddPass(std::make_shared<ResizeV2Fusion>());
  common_pm->AddPass(std::make_shared<SparseConcatFusion>());
  common_pm->AddPass(std::make_shared<SparseCrossFusion>());
  common_pm->AddPass(std::make_shared<SparseTensorDenseMatMulFusion>());
  common_pm->AddPass(std::make_shared<SplitFusion>());
  common_pm->AddPass(std::make_shared<StandardNormalFusion>());
  common_pm->AddPass(std::make_shared<Conv3DBackpropInputPadListFusion>());
  common_pm->AddPass(std::make_shared<Conv3DBackpropFilterPadListFusion>());
  return common_pm;
}

void BackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  PROF_START(backend_common_optimization);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start common optimization. graph id: " << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  optimizer->AddPassManager(GetBackendCommonOptimizationPassManagerPtr(kernel_graph));
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  PROF_END(backend_common_optimization);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end common optimization. graph id: " << kernel_graph->graph_id();
}

// Delete this optimizer when dynamic and static ReduceSum is unified.
void OpBackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Status record: start op common optimization. graph id: " << kernel_graph->graph_id();
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto common_pm = std::make_shared<PassManager>("op_common_pm");
  common_pm->AddPass(std::make_shared<ReduceOptimizer>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<BroadcastToFusion>());
  common_pm->AddPass(std::make_shared<AccumulateNV2Fusion>());
  common_pm->AddPass(std::make_shared<AddNFusion>());
  common_pm->AddPass(std::make_shared<ArgMaxWithValueFusion>());
  common_pm->AddPass(std::make_shared<ArgMinWithValueFusion>());
  common_pm->AddPass(std::make_shared<BatchMatMulAttrFusion>());
  common_pm->AddPass(std::make_shared<ConcatOffsetV1Fusion>());
  common_pm->AddPass(std::make_shared<DynamicRNNFusion>());
  common_pm->AddPass(std::make_shared<GatherFusion>());
  common_pm->AddPass(std::make_shared<Im2ColFusion>());
  common_pm->AddPass(std::make_shared<IOUFusion>());
  common_pm->AddPass(std::make_shared<LogFusion>());
  common_pm->AddPass(std::make_shared<MaxPoolWithArgmaxV2Fusion>());
  common_pm->AddPass(std::make_shared<NanToNumFusion>());
  common_pm->AddPass(std::make_shared<ParallelConcatFusion>());
  common_pm->AddPass(std::make_shared<RaggedTensorToSparseFusion>());
  common_pm->AddPass(std::make_shared<ResizeV2Fusion>());
  common_pm->AddPass(std::make_shared<SparseConcatFusion>());
  common_pm->AddPass(std::make_shared<SparseCrossFusion>());
  common_pm->AddPass(std::make_shared<SparseTensorDenseMatMulFusion>());
  common_pm->AddPass(std::make_shared<SplitFusion>());
  common_pm->AddPass(std::make_shared<StandardNormalFusion>());
  common_pm->AddPass(std::make_shared<Conv3DBackpropInputPadListFusion>());
  common_pm->AddPass(std::make_shared<Conv3DBackpropFilterPadListFusion>());
  if (kernel_graph->has_attr(kAttrPackFunction)) {
    common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  }
  optimizer->AddPassManager(common_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  MS_LOG(INFO) << "Status record: end op common optimization. graph id: " << kernel_graph->graph_id();
}

void CommonFinalOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // Run optimizer passes.
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("final_opt");
  pm->AddPass(std::make_shared<OptimizeUpdateState>());
  pm->AddPass(std::make_shared<AddAkgKernelAttrs>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  // Dump IR if save_graphs is set.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    std::string filename = "hwopt_common_final_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(filename, kernel_graph);
  }
  std::string device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  int execution_mode = context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (device_target != kAscendDevice || execution_mode != kPynativeMode) {
    // Here dump graphs only for Ascend pynative mode.
    return;
  }
  std::string final_graph = "trace_code_graph_" + std::to_string(kernel_graph->graph_id());
  auto &json_parser = DumpJsonParser::GetInstance();
  if (json_parser.e2e_dump_enabled() || json_parser.async_dump_enabled()) {
    uint32_t rank_id = GetRankId();
    std::string root_dir = json_parser.path() + "/rank_" + std::to_string(rank_id);
    MS_LOG(INFO) << "Dump graph and exeorder for graph: " << kernel_graph->graph_id()
                 << "root_graph_id: " << kernel_graph->root_graph_id();
    std::string target_dir = root_dir + "/graphs";
    std::string cst_file_dir = GenerateDumpPath(kernel_graph->root_graph_id(), rank_id, true);
    std::string ir_file_path = target_dir + "/" + "ms_output_" + final_graph + ".ir";
    DumpIRProtoWithSrcInfo(kernel_graph, final_graph, target_dir, kDebugWholeStack);
    if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      // Dump constant data for old runtime ascend.
      DumpConstantInfo(kernel_graph, cst_file_dir);
    }
    DumpIR("trace_code_graph", kernel_graph, true, kWholeStack, ir_file_path);
    DumpGraphExeOrder("ms_execution_order_graph_" + std::to_string(kernel_graph->graph_id()) + ".csv", root_dir,
                      kernel_graph->execution_order());
  }
#endif
}

PassManagerPtr GetCommonUnifyMindIRPassManager() {
  auto pm = std::make_shared<PassManager>("common_unify_mindir_pm");
  pm->AddPass(std::make_shared<ConvTransposeToConvBackpropInputPass>());
  pm->AddPass(std::make_shared<CustomOpRegInfoToAttr>());
  pm->AddPass(std::make_shared<InplaceAssignForCustomOp>());
  pm->AddPass(std::make_shared<ConvertAttrToUnifyMindIR>());
  return pm;
}

void CommonUnifyMindIR(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
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
  auto opt = std::make_shared<GraphOptimizer>();
  opt->AddPassManager(GetCommonUnifyMindIRPassManager());
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_common_unify_mindir_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void AddDynamicShapeAttrPass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("add_dynamic_shape_attr");
  pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
}

PassManagerPtr GetEliminateIllegalDataTypePassManager() {
  auto pm = std::make_shared<PassManager>("common_eliminate_illegal_data_type_pm");
  pm->AddPass(std::make_shared<ConvertListToTuple>("convert_list_to_tuple"));
  pm->AddPass(std::make_shared<EliminateFuncDataType>());
  return pm;
}

void EliminateIllegalDataTypePass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
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
  auto opt = std::make_shared<GraphOptimizer>();
  opt->AddPassManager(GetEliminateIllegalDataTypePassManager());
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void DynamicShapeConvertPass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start dynamic shape convert for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_dynamic_shape_convert_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto dynamic_shape_convert_pm = std::make_shared<opt::PassManager>("dynamic_shape_convert_pm");
  dynamic_shape_convert_pm->AddPass(std::make_shared<opt::dynamic_shape::ConvertCustomOp>());
  dynamic_shape_convert_pm->AddPass(std::make_shared<opt::dynamic_shape::LinkCustomOp>());
  optimizer->AddPassManager(dynamic_shape_convert_pm);
  (void)optimizer->Optimize(kernel_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_dynamic_shape_convert_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void OptimizationWithoutBackend(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
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
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_optimization_without_backend_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void OptimizationForAnyTypeKernelGraph(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto common_pm = std::make_shared<opt::PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<ConvertListToTuple>("convert_list_to_tuple"));
  common_pm->AddPass(std::make_shared<EliminateFuncDataType>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<opt::ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<opt::ConvertTupleOutputToMaketuple>());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(common_pm);
  optimizer->Optimize(kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
