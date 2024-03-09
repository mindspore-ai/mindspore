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

#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"

#include <memory>
#include <string>
#include "backend/common/pass/dropout_gen_mask_fusion.h"
#include "backend/common/pass/common_subexpression_elimination.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/optimizer/ge/all_to_all_v_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/maketuple_depend_remover.h"
#include "plugin/device/ascend/optimizer/ge/expand_dims_for_batchnorm.h"
#include "plugin/device/ascend/optimizer/ge/convert_data_depend_to_control_depend.h"
#include "plugin/device/ascend/optimizer/ge/convert_condition_input_to_scalar.h"
#include "plugin/device/ascend/optimizer/ge/hcom/add_parallel_group_for_hcom.h"
#include "plugin/device/ascend/optimizer/ge/hcom/insert_tensor_move_for_hccl_op_ge.h"
#include "plugin/device/ascend/optimizer/ge/hcom/insert_depend_for_all_gather_ge.h"
#include "plugin/device/ascend/optimizer/ge/trans_depend_value_to_int32.h"
#include "plugin/device/ascend/optimizer/ge/expander_fallback.h"
#include "plugin/device/ascend/optimizer/ge/insert_identity.h"
#include "plugin/device/ascend/optimizer/ge/dropout_gen_mask_depend.h"
#include "plugin/device/ascend/optimizer/ge/unfold_maketuple.h"
#include "plugin/device/ascend/optimizer/ge/unfold_nested_output.h"
#include "plugin/device/ascend/optimizer/ge/resize_bilinear_add_attr.h"
#include "plugin/device/ascend/optimizer/ge/process_call_inline.h"
#include "plugin/device/ascend/optimizer/ge/process_partial_inline.h"
#include "plugin/device/ascend/optimizer/format_type/deal_ref_output.h"
#include "plugin/device/ascend/optimizer/ge/hcom/insert_load_for_allgather.h"
#include "plugin/device/ascend/optimizer/format_type/set_fracz_group_attr.h"
#include "plugin/device/ascend/optimizer/ge/shape_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/inputs_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/maketuple_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/add_cast_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/scalar_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/tuple_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ir_fission/seed_adapter.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/remove_tensor_to_scalar_or_tuple_ops.h"
#include "plugin/device/ascend/optimizer/ge/scalar_ops_output_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/ge_convert_const_input_to_tensor_input.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "plugin/device/ascend/optimizer/enhancer/eliminate_maketuple_getitem.h"

namespace mindspore {
namespace opt {
void GEBackendOptimization(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
  PROF_START(ascend_backend_optimize_ge);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_ge_pm = std::make_shared<PassManager>("opt_ge_pm");
  opt_ge_pm->AddPass(std::make_shared<opt::GEConvertConstInputToTensorInput>());
  opt_ge_pm->AddPass(std::make_shared<opt::RemoveTensorToScalarOrTupleOps>());
  opt_ge_pm->AddPass(std::make_shared<opt::AllToAllvForGE>());
  opt_ge_pm->AddPass(std::make_shared<opt::InsertLoadForAllGather>());
  opt_ge_pm->AddPass(std::make_shared<opt::InsertTensorMoveForHcclOpGe>());
  opt_ge_pm->AddPass(std::make_shared<opt::InsertDependForAllGatherGe>());
  opt_ge_pm->AddPass(std::make_shared<opt::ConvertCondInputToScalar>());
  opt_ge_pm->AddPass(std::make_shared<opt::ConvertDataDependToControlDepend>());
  opt_ge_pm->AddPass(std::make_shared<opt::MakeTupleDependRemover>());
  opt_ge_pm->AddPass(std::make_shared<opt::AddParallelGroupForHcom>());
  opt_ge_pm->AddPass(std::make_shared<opt::ExpandDimsForBatchNorm>());
  opt_ge_pm->AddPass(std::make_shared<opt::DropoutGenMaskDepend>());
  opt_ge_pm->AddPass(std::make_shared<opt::AddCastForGe>());
  opt_ge_pm->AddPass(std::make_shared<opt::ResizeBilinearAddAttr>());
  opt_ge_pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>(true, true));
  opt_ge_pm->AddPass(std::make_shared<opt::UnfoldNestedOutput>("unfold_nested_output"));
  opt_ge_pm->AddPass(std::make_shared<opt::UnfoldMaketuple>("unfold_nested_maketuple"));
  optimizer->AddPassManager(opt_ge_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_ge_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  PROF_END(ascend_backend_optimize_ge);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize ge pass. graph id: " << kernel_graph->graph_id();
}

void GEBackendOptimizeACL(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 0);
  PROF_START(ascend_backend_optimize_acl);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_pm = std::make_shared<PassManager>("opt_acl_pm");
  opt_acl_pm->AddPass(std::make_shared<SeedAdapter>());
  opt_acl_pm->AddPass(std::make_shared<InsertTensorMoveForCommunication>());
  opt_acl_pm->AddPass(std::make_shared<opt::TransDependValueToInt32>());
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessCallInline>());
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessPartialInline>());
  opt_acl_pm->AddPass(std::make_shared<opt::ExpanderFallback>());
  optimizer->AddPassManager(opt_acl_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_opt_acl_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  PROF_END(ascend_backend_optimize_acl);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 1);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void GEBackendOptimizeACLAfterKernelSelect(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel select. graph id: "
                << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelSelect", 0, 0,
                            0);
  PROF_START(ascend_backend_optimize_acl_after_kernel_select);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_opt_acl_graph_after_kernel_select_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_after_kernel_select_pm = std::make_shared<PassManager>("opt_acl_after_kernel_select_pm");
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<SetFraczGroupAttr>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<InsertIdentity>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<EraseVisitAttr>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<DealRefOutput>());
  if (!kernel_graph->is_from_single_op() && !kernel_graph->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  }
  optimizer->AddPassManager(opt_acl_after_kernel_select_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_end_opt_acl_graph_after_kernel_select_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  PROF_END(ascend_backend_optimize_acl_after_kernel_select);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelSelect", 0, 0,
                            1);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void GEUnifyMindIR(const KernelGraphPtr &kernel_graph) {
  profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_UnifyMindIR", 0, 0, 0);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
    DumpIRProto(kernel_graph, "before_unify_mindir_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(GetGEUnifyMindIRPassManager());
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_UnifyMindIR", 0, 0, 1);
}

void GEAfterInlineOptimize(const KernelGraphPtr &kernel_graph) {
  profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_AfterInline", 0, 0, 0);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_inline_optimize_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto after_inline_pm = std::make_shared<PassManager>("after_inline_pm");
  after_inline_pm->AddPass(std::make_shared<DropoutGenMaskFusion>());
  after_inline_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  after_inline_pm->AddPass(std::make_shared<EliminateMaketupleGetitem>());
  optimizer->AddPassManager(after_inline_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_inline_optimize_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  profiler::CollectHostInfo("GE", "Graph Optimization", "BackendOptimization_AfterInline", 0, 0, 1);
}

void GEDynamicUnifyMindIR(const FuncGraphPtr &func_graph) {
  profiler::CollectHostInfo("GE", "GE Dynamic Shape Unify MindIR", "GEBackend_Dynamic_UnifyMindIR", 0, 0, 0);
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
  auto dynamic_unify_mindir_pm = std::make_shared<opt::PassManager>("ge_dynamic_unify_mindir_pm");
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::ScalarOpsOutputUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::ShapeUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::MakeTupleUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::InputsUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::ScalarUnifyMindIR>());
  dynamic_unify_mindir_pm->AddPass(std::make_shared<opt::TupleUnifyMindIR>());
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  optimizer->AddPassManager(dynamic_unify_mindir_pm);
  (void)optimizer->Optimize(func_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_ge_dynamic_shape_unify_mindir_graph.ir";
    DumpIR(file_name, func_graph);
  }
#endif
  profiler::CollectHostInfo("GE", "GE Dynamic Shape Unify MindIR", "GEBackend_Dynamic_UnifyMindIR", 0, 0, 1);
}

PassManagerPtr GetGEUnifyMindIRPassManager() {
  auto unify_mindir_pm = std::make_shared<opt::PassManager>("ge_unify_mindir_pm");
  MS_EXCEPTION_IF_NULL(unify_mindir_pm);
  GetBackendCommonUnifyMindIRPassManager(&unify_mindir_pm);
  return unify_mindir_pm;
}
}  // namespace opt
}  // namespace mindspore
