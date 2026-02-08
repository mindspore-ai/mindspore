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

#include "plugin/ascend/graph_optimizer/pass/ascend_pass_optimization.h"

#include <memory>
#include <string>

#include "backend/common/pass/dropout_gen_mask_fusion.h"
#include "backend/common/pass/common_subexpression_elimination.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "backend/common/pass/concat_outputs_for_all_gather.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "backend/common/pass/graph_view_replace_pass.h"
#include "backend/common/pass/insert_type_transform_op.h"
#include "backend/common/pass/insert_tensor_move_for_communication.h"
#include "backend/common/pass/split_inputs_for_reduce_scatter.h"
#include "backend/common/pass/label_1f1b_overlap_node.h"
#include "backend/common/pass/overlap_grad_reduce.h"
#include "backend/common/pass/overlap_1b1f.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/utils/parallel_context.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "tools/profiler/profiling.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "plugin/ascend/graph_optimizer/pass/backend_common_unify_mindir.h"
#include "plugin/ascend/graph_optimizer/pass/enhancer/eliminate_maketuple_getitem.h"
#include "plugin/ascend/graph_optimizer/pass/format_type/deal_ref_output.h"
#include "plugin/ascend/graph_optimizer/pass/format_type/set_fracz_group_attr.h"
#include "plugin/ascend/graph_optimizer/pass/expander_fallback.h"
#include "plugin/ascend/graph_optimizer/pass/format_type/insert_identity.h"
#include "plugin/ascend/graph_optimizer/pass/format_type/format_cast_modify_output.h"
#include "plugin/ascend/graph_optimizer/pass/heterogeneous/insert_pre_fetch_depend.h"
#include "backend/common/pass/other/process_call_inline.h"
#include "backend/common/pass/other/process_partial_inline.h"
#include "backend/common/pass/other/convert_pad_v3_paddings.h"
#include "plugin/ascend/graph_optimizer/pass/heterogeneous/insert_move_to.h"
#include "backend/common/pass/ir_fission/seed_adapter.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/shape_reshape_fusion.h"
#include "backend/common/pass/other/hcom/insert_tensor_move_for_hccl_op_ge.h"
#include "backend/common/pass/other/resize_bilinear_add_attr.h"
#include "backend/common/pass/custom_defined_depend.h"

namespace mindspore {
namespace opt {
void AclAfterCreateKernel(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel create. graph id: "
                << kernel_graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 0);
  PROF_START(AclAfterCreateKernel);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_acl_graph_final_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_ack = std::make_shared<PassManager>("opt_acl_ack");
  opt_acl_ack->AddPass(std::make_shared<EraseVisitAttr>());
  opt_acl_ack->AddPass(std::make_shared<DealRefOutput>());
  optimizer->AddPassManager(opt_acl_ack);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AclAfterCreateKernel);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_end_acl_graph_final_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACL", 0, 0, 1);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass after kernel create. graph id: "
                << kernel_graph->graph_id();
}

void AscendGraphOptimizeACL(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(AscendGraphOptimizeACL);
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
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessCallInline>());
  opt_acl_pm->AddPass(std::make_shared<SeedAdapter>());

  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeInsertTensorMove)) {
    opt_acl_pm->AddPass(std::make_shared<opt::InsertTensorMoveForHcclOpGe>());
  } else {
    opt_acl_pm->AddPass(std::make_shared<InsertTensorMoveForCommunication>());
  }
  opt_acl_pm->AddPass(std::make_shared<opt::ProcessPartialInline>());
  opt_acl_pm->AddPass(std::make_shared<opt::ExpanderFallback>());
  opt_acl_pm->AddPass(std::make_shared<opt::ConvertPadV3Paddings>());
  opt_acl_pm->AddPass(std::make_shared<opt::ConvertPadV3GradPaddings>());
  opt_acl_pm->AddPass(std::make_shared<opt::ResizeBilinearAddAttr>());
  opt_acl_pm->AddPass(std::make_shared<opt::CustomDefinedDepend>(false, kernel_graph->graph_id()));
  optimizer->AddPassManager(opt_acl_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AscendGraphOptimizeACL);
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

void AscendGraphOptimizeACLAfterKernelSelect(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel select. graph id: "
                << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(AscendGraphOptimizeACLAfterKernelSelect);
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
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<FormatCastModifyOutput>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<InsertIdentity>());

  // graph_mode process the pass in OptimizeACLGraphAfterCreateKernel
  if (!IsJit()) {
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<EraseVisitAttr>());
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<DealRefOutput>());
  }

  if (!kernel_graph->is_from_single_op()) {
    opt_acl_after_kernel_select_pm->AddPass(std::make_shared<opt::InsertTypeTransformOp>());
  }
  if (!kernel_graph->is_graph_run_mode() && context_ptr->ascend_soc_version() != "ascend910") {
    bool infer_boost = context_ptr->IsEnableInferBoost();
    opt_acl_after_kernel_select_pm->AddFusionPass(std::make_shared<opt::ShapeReshapeFusion>(), infer_boost);
    opt_acl_after_kernel_select_pm->AddFusionPass(std::make_shared<opt::ShapeReshapeDirectFusion>());
  }
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<opt::GraphViewReplacePass>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<Label1F1BOverlapNode>());
  opt_acl_after_kernel_select_pm->AddPass(std::make_shared<InsertMoveTo>());
  optimizer->AddPassManager(opt_acl_after_kernel_select_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AscendGraphOptimizeACLAfterKernelSelect);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_end_opt_acl_graph_after_kernel_select_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelSelect",
                                  start_time, profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void AscendGraphOptimizeACLAfterKernelPacket(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ge_mode = mindspore::AnfAlgo::GetBackend(kernel_graph) == kBackendGE;
  if (is_ge_mode || context_ptr->IsEnableInferBoost() || kernel_graph->is_from_single_op()) {
    return;
  }

  MS_LOG(DEBUG) << "Status record: start ascend backend optimize acl pass after kernel packet. graph id: "
                << kernel_graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(AscendGraphOptimizeACLAfterKernelPacket);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_opt_acl_graph_after_kernel_packet_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto opt_acl_after_kernel_packet_pm = std::make_shared<PassManager>("opt_acl_after_kernel_packet");
  opt_acl_after_kernel_packet_pm->AddPass(std::make_shared<opt::GraphViewReplacePass>());
  PROF_END(AscendGraphOptimizeACLAfterKernelPacket);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_end_opt_acl_graph_after_kernel_packet_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  optimizer->AddPassManager(opt_acl_after_kernel_packet_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_OptimizeACLAfterKernelPacket",
                                  start_time, profiler::GetClockSyscnt(), 0);
  MS_LOG(DEBUG) << "Status record: end ascend backend optimize acl pass. graph id: " << kernel_graph->graph_id();
}

void AscendUnifyMindIR(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(AscendUnifyMindIR);
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
  optimizer->AddPassManager(GetBackendCommonUnifyMindIRPassManager());
  optimizer->AddPassManager(GetBackendFusionGroupPassManager());
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AscendUnifyMindIR);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_unify_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnifyMindIR", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void AscendAfterInlineOptimize(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  PROF_START(AscendAfterInlineOptimize);
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
  after_inline_pm->AddFusionPass(std::make_shared<DropoutGenMaskFusion>());
  after_inline_pm->AddPass(std::make_shared<CommonSubexpressionElimination>());
  after_inline_pm->AddPass(std::make_shared<EliminateMaketupleGetitem>());
  after_inline_pm->AddPass(std::make_shared<InsertPreFetchDepend>());
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Check whether the GPTO option "passes" is set to be off
  // Under GPTO mode, this option is to disable manual overlap passes
  const auto &backend_jit_config = kernel_graph->backend_jit_config();
  if (backend_jit_config.IsGptoPassesEnabled()) {
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_GRAD_COMM_OPT)) {
      after_inline_pm->AddPass(std::make_shared<OverlapGradReduce>());
    }
    after_inline_pm->AddPass(std::make_shared<Overlap1b1f>());
  }
  optimizer->AddPassManager(after_inline_pm);
  (void)optimizer->Optimize(kernel_graph);
  PROF_END(AscendAfterInlineOptimize);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_inline_optimize_mindir_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_AfterInline", start_time,
                                  profiler::GetClockSyscnt(), 0);
}
}  // namespace opt
}  // namespace mindspore
