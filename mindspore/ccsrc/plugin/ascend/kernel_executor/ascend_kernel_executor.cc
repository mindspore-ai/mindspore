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
#include "plugin/ascend/kernel_executor/ascend_kernel_executor.h"

#include <algorithm>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ir/tensor_new.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"
#include "include/utils/parallel_context.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "ir/graph_utils.h"
#include "tools/profiler/profiler.h"
#include "include/utils/utils.h"
#include "mindapi/base/type_id.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "include/backend/backend_manager/backend_jit_config.h"
#include "include/backend/common/kernel_graph/kernel_graph_mgr.h"
#include "plugin/ascend/res_manager/device_context_conf/op_debug_conf.h"
#include "plugin/ascend/res_manager/device_context_conf/op_precision_conf.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "plugin/ascend/graph_optimizer/ascend_graph_optimization.h"
#include "plugin/ascend/graph_optimizer/somas/acl_somas.h"
#include "plugin/ascend/graph_optimizer/stream_assign/acl_stream_assign.h"
#include "plugin/ascend/stress_detect/stress_detect.h"
#include "plugin/ascend/kernel_executor/rts/rt_kernel_build.h"
#include "mindspore/ccsrc/plugin/ascend/kernel_executor/hyper_offload/hyper_offload.h"
#include "kernel/ascend/hccl/hccl_kernel_metadata.h"
#include "kernel/ascend/hccl/hccl_kernel_build.h"
#include "kernel/ascend/simu/kernel_mod_impl/simu_kernel_build.h"
#include "kernel/ascend/internal/internal_kernel_build.h"
#include "kernel/ascend/symmetric_memory/symmetric_memory_kernel_build.h"
#include "kernel/ascend/custom/kernel_mod_impl/custom_kernel_build.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "kernel/ascend/kernel_packet/kernel_mod_impl/kernel_packet_ascend_kernel_mod.h"
#include "plugin/ascend/res_manager/mbuf_manager/tensorreport_utils.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#ifdef ENABLE_DVM
#include "kernel/ascend/dvm/kernel_mod_impl/dvm_kernel_build.h"
#endif

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/backend/common/pass_manager/helper.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_build.h"
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_build.h"
#include "kernel/ascend/atb/kernel_mod_impl/atb_kernel_build.h"
#include "mindspore/ops/kernel/host/host_kernel_build.h"
#include "plugin/ascend/kernel_executor/host/host_kernel_metadata.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_util.h"
#include "kernel/ascend/acl_ir/ge_adapter_info.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "plugin/ascend/res_manager/ascend_res_manager.h"
#include "kernel/ascend/aclop/kernel_mod_impl/acl_kernel_mod.h"
#include "plugin/ascend/ascend_device_context.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::device::ascend {
namespace {
constexpr size_t kSwitchInputSize = 3;
constexpr size_t kSwitchCondIndex = 1;
constexpr size_t kSwitchBranchTrueIndex = 2;
constexpr size_t kSwitchBranchFalseIndex = 3;

std::string GetKernelTypeStr(const KernelType &kernel_type) {
  std::string type = "";
  if (kernel_type == KernelType::ACL_KERNEL) {
    type = "acl_kernel";
  } else if (kernel_type == KernelType::HOST_KERNEL) {
    type = "host_kernel";
  } else if (kernel_type == KernelType::HCCL_KERNEL) {
    type = "hccl_kernel";
  } else if (kernel_type == KernelType::OPAPI_KERNEL) {
    type = "opapi_kernel";
  } else if (kernel_type == KernelType::INTERNAL_KERNEL) {
    type = "internal_kernel";
  } else if (kernel_type == KernelType::AKG_KERNEL) {
    type = "akg_kernel";
  } else if (kernel_type == KernelType::RT_KERNEL) {
    type = "rt_kernel";
  } else if (kernel_type == KernelType::ATB_KERNEL) {
    type = "atb_kernel";
  }
  return type;
}

kernel::KernelModPtr GenerateAkgKernelMod(const CNodePtr &kernel);

bool GenerateKernelMod(const std::vector<CNodePtr> &kernels) {
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelMod(kernel)) {
      continue;
    }
    if (AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    std::string opname = common::AnfAlgo::GetCNodeName(kernel);
    kernel::KernelModPtr kernel_mod_ptr = nullptr;
    auto kernel_type = AnfAlgo::GetKernelType(kernel);
    if (kernel_type == KernelType::ACL_KERNEL) {
      kernel_mod_ptr = kernel::AclOpBuild(kernel);
    } else if (kernel_type == KernelType::HOST_KERNEL) {
      kernel_mod_ptr = kernel::HostOpBuild(kernel);
    } else if (kernel_type == KernelType::HCCL_KERNEL) {
      if (common::IsExecuteSimulation()) {
        kernel_mod_ptr = kernel::SimuOpBuild(kernel);
      } else {
        kernel_mod_ptr = kernel::HcclOpBuild(kernel);
      }
    } else if (kernel_type == KernelType::OPAPI_KERNEL) {
      kernel_mod_ptr = kernel::AclnnOpBuild(kernel);
    } else if (kernel_type == KernelType::AKG_KERNEL) {
      kernel_mod_ptr = GenerateAkgKernelMod(kernel);
    } else if (kernel_type == KernelType::RT_KERNEL) {
      kernel_mod_ptr = kernel::RtOpBuild(kernel);
    } else if (kernel_type == KernelType::INTERNAL_KERNEL) {
      kernel_mod_ptr = kernel::InternalKernelBuild(kernel);
    } else if (kernel_type == KernelType::SYMMETRIC_MEMORY_KERNEL) {
      kernel_mod_ptr = kernel::SymmetricMemoryKernelBuild(kernel);
    } else if (kernel_type == KernelType::ATB_KERNEL) {
      kernel_mod_ptr = kernel::AtbKernelBuild(kernel);
    } else if (kernel_type == KernelType::CUSTOM_KERNEL) {
      kernel_mod_ptr = kernel::CustomKernelBuild(kernel);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, kernel)
        << "The kernel: " << kernel->fullname_with_scope()
        << " kernel build failed, kernel type: " << kernel::KernelTypeLabel(AnfAlgo::GetKernelType(kernel));
    }
    MS_LOG(INFO) << "kernel opname:" << opname << ", kernel type:" << GetKernelTypeStr(kernel_type);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, kernel.get());
  }
  return true;
}

void ExecKernelModResize(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = kernel_graph->execution_order();
  for (const auto &kernel : nodes) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto cnode = kernel->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto kernel_type = AnfAlgo::GetKernelType(kernel);
    std::string opname = common::AnfAlgo::GetCNodeName(kernel);
    MS_LOG(DEBUG) << "kernel opname:" << opname << ", kernel type:" << GetKernelTypeStr(kernel_type);
    if (kernel_type == KernelType::OPAPI_KERNEL) {
      auto prim = kernel_mod->primitive();
      prim->set_attr(kAttrStreamId, MakeValue<uint32_t>(AnfAlgo::GetStreamId(kernel)));
    }
    if (kernel::CheckResizeCondition(cnode)) {
      std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
      std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);
      if (kernel_mod->Resize(input_kernel_tensors, output_kernel_tensors) == kernel::KRET_RESIZE_FAILED) {
        MS_LOG(EXCEPTION) << "#dmsg#Kernel resize failed:#dmsg#internal kernel op[" << cnode->fullname_with_scope()
                          << "] Resize failed.";
      }
    }
  }
}

kernel::KernelModPtr CreateKernelPacketKernelMod(const CNodePtr &kernel) {
  MS_LOG(DEBUG) << "Build KernelPacket: " << kernel->DebugString();
  auto real_kernel = kernel::GetKernelPacketRealNode(kernel);
  if (!GenerateKernelMod({real_kernel})) {
    MS_LOG(ERROR) << "Build " << real_kernel->DebugString() << " failed.";
    return nullptr;
  }
  auto kp_kernelmod = std::make_shared<kernel::KernelPacketAscendKernelMod>();
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel);
  MS_EXCEPTION_IF_NULL(prim);
  // when the kernel is inlined, multiple kernels may share a Primitive, so clone an object.
  prim = prim->Clone();
  kernel->set_input(0, NewValueNode(prim));
  auto name = GetValue<std::string>(prim->GetAttr(kAttrKernelPacketNode));
  auto infer_func = std::make_shared<kernel::KernelPacketInfer>(name, real_kernel->func_graph(), kp_kernelmod.get());
  prim->set_attr("infer_shape_functor", infer_func);
  auto real_kernel_info = dynamic_cast<device::KernelInfo *>(real_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(real_kernel_info);
  if (!kp_kernelmod->KernelMod::Init(prim, input_kernel_tensors, output_kernel_tensors) ||
      !kernel::KernelPacketInitializer::InitKernel(real_kernel, real_kernel_info->GetKernelMod(), kp_kernelmod.get(),
                                                   infer_func.get())) {
    MS_LOG_WITH_NODE(EXCEPTION, real_kernel)
      << "#dmsg#Kernel build failed:#dmsg#Initialize kernel op[" << real_kernel->fullname_with_scope() << "] failed.";
  }
  real_kernel_info->set_kernel_mod(nullptr);
  return kp_kernelmod;
}

kernel::KernelModPtr GenerateAkgKernelMod(const CNodePtr &kernel) {
  if (common::AnfAlgo::HasNodeAttr(kAttrKernelPacketNode, kernel)) {
    return CreateKernelPacketKernelMod(kernel);
  }
#ifdef ENABLE_DVM
  return kernel::DvmOpBuild(kernel);
#else
  return nullptr;
#endif
}

void SetAclOpPrecisionMode() {
  auto op_precision_conf = OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);

  auto precision_mode = op_precision_conf->precision_mode();
  if (precision_mode.empty()) {
    precision_mode =
      (device::ascend::AclUtil::KeepOriginDType() == 1) ? "must_keep_origin_dtype" : "allow_fp32_to_fp16";
  }
  MS_LOG(INFO) << "Set aclop PRECISION_MODE: " << precision_mode;
  auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_PRECISION_MODE, precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set precision mode failed! Error flag is " << ret;
  }

  auto op_precision_mode = op_precision_conf->op_precision_mode();
  if (op_precision_mode.empty()) {
    return;
  }
  MS_LOG(INFO) << "Set aclop OP_PRECISION_MODE: " << op_precision_mode;
  ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_PRECISION_MODE, op_precision_mode.c_str());
  if (ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Acl set op precision mode failed! Error flag is " << ret;
  }
}

void SelectKernelInfo(const KernelGraphPtr &kernel_graph, const CNodePtr &kernel, bool *is_aclop = nullptr,
                      std::vector<size_t> *op_selected_num = nullptr) {
  auto [select_res, msg, etype, is_aclop_local] =
    device::ascend::SelectKernelInfoWithMsg(kernel_graph, kernel, op_selected_num);
  if (is_aclop != nullptr) {
    *is_aclop = is_aclop_local;
  }
  if (!select_res) {
    MS_LOG(INFO) << "node is " << kernel->fullname_with_scope() << " should backoff";
    std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
    device::ascend::HandleKernelSelectFailure(kernel_graph, kernel, failure_info);
  }
}

void SelectKernel(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *const memo,
                  std::vector<size_t> *op_selected_num, bool *has_aclop = nullptr) {
  // select kernel
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(SelectKernel);
  if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  memo->insert(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  const auto &kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
    bool is_aclop = false;
    SelectKernelInfo(kernel_graph, kernel, &is_aclop, op_selected_num);
    if (has_aclop != nullptr) {
      // cppcheck-suppress useStlAlgorithm
      *has_aclop |= is_aclop;
    }
  }
  if (!kernel_graph->is_from_single_op()) {
    kernel_graph->SetKernelObjectTypesForUnrealNodes();
  }
  for (auto &child_graph : kernel_graph->child_graph_order()) {
    SelectKernel(child_graph.lock(), memo, op_selected_num);
  }
  PROF_END(SelectKernel);
}

inline void PrintOpSelectedNum(const std::vector<size_t> &op_selected_num) {
  MS_LOG(INFO) << "Number of INTERNAL_KERNEL, OPAPI_KERNEL, ACL_KERNEL, ATB_KERNEL, "
               << "HCCL_KERNEL, HOST_KERNEL, SYMMETRIC_MEMORY_KERNEL:";
  MS_VLOG(VL_FLOW) << "Number of INTERNAL_KERNEL, OPAPI_KERNEL, ACL_KERNEL, ATB_KERNEL, "
                   << "HCCL_KERNEL, HOST_KERNEL, SYMMETRIC_MEMORY_KERNEL:";
  std::stringstream ss;
  for (const auto num : op_selected_num) {
    ss << num << " ";
  }
  MS_LOG(INFO) << ss.str();
  MS_VLOG(VL_FLOW) << ss.str();
}

void InlineSubGraph(const KernelGraphPtr &graph, const KernelGraphPtr &sub_graph, CNodePtr kernel_cnode,
                    AnfNodePtr *last_call, AnfNodePtr *pre_last_call, bool is_switch_inline) {
  MS_EXCEPTION_IF_NULL(kernel_cnode);
  MS_EXCEPTION_IF_NULL(sub_graph);
  MS_LOG(INFO) << "InlineSubGraph: " << kernel_cnode->fullname_with_scope() << ", sub graph: " << sub_graph->graph_id()
               << ", need inline: " << sub_graph->need_inline();
  auto main_graph = kernel_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(main_graph);
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = MakeManager({main_graph});
    MS_EXCEPTION_IF_NULL(mng);
    mng->AddFuncGraph(main_graph);
    main_graph->set_manager(mng);
  }
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_cnode->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  AnfNodePtrList inp;
  auto &call_input = kernel_cnode->inputs();
  // let operators on different subgraphs will not be executed interleavedly
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto pp_1f1b_value = ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  for (size_t i = 1; i < call_input.size(); i++) {
    if ((pp_1f1b_value.empty() || !kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp)) && last_call != nullptr &&
        (*last_call) != nullptr) {
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), call_input[i], (*last_call)});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(call_input[i]->abstract());
      depend->AddAttr("inline_depend", MakeValue(true));
      if (!pp_1f1b_value.empty() && pre_last_call != nullptr && (*pre_last_call) != nullptr) {
        auto depend_new = graph->NewCNode({NewValueNode(prim::kPrimDepend), depend, (*pre_last_call)});
        depend_new->set_abstract(call_input[i]->abstract());
        depend_new->AddAttr("inline_depend2", MakeValue(true));
        inp.push_back(depend_new);
      } else {
        inp.push_back(depend);
      }
    } else if (kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp) && pre_last_call != nullptr &&
               (*pre_last_call) != nullptr) {
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), call_input[i], (*pre_last_call)});
      depend->set_abstract(call_input[i]->abstract());
      depend->AddAttr("inline_depend1", MakeValue(true));
      inp.push_back(depend);
    } else {
      inp.push_back(call_input[i]);
    }
  }
  if (pre_last_call != nullptr && last_call != nullptr) {
    (*pre_last_call) = (*last_call);
  }
  const auto &ref_map = sub_graph->GetRefMap();
  auto out = session::KernelGraphMgr::DoInline(sub_graph, main_graph, inp, kernel_cnode, kernel_info->graph_id(),
                                               ref_map, graph, is_switch_inline);
  (void)mng->Replace(kernel_cnode, out);
  // Inline graph boundary: MakeTuple---->Depend---->Tensormove
  // Avoid long link times at runtime
  if (last_call != nullptr) {
    static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
    if (!enable_infer_boost) {
      auto value_node = graph->NewValueNode(MakeValue(tensor::from_scalar(1)));
      MS_EXCEPTION_IF_NULL(value_node);
      CNodePtr tensor_move = nullptr;
      auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), value_node, out});
      MS_EXCEPTION_IF_NULL(depend);
      depend->set_abstract(value_node->abstract());
      tensor_move = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTensorMove->name())), depend});
      MS_EXCEPTION_IF_NULL(tensor_move);
      tensor_move->set_abstract(value_node->abstract());
      common::AnfAlgo::SetNodeAttr(kAttrKernelGraphBoundary, MakeValue(sub_graph->ToString()), tensor_move);
      // select kernel
      SelectKernelInfo(graph, tensor_move);
      (*last_call) = tensor_move;
    } else {
      (*last_call) = out;
    }
  }
}

void InlineCallGraph(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  PROF_START(InlineCallGraph);
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  graph->SetExecOrderByDefault();
  auto kernel_cnodes = graph->execution_order();
  AnfNodePtr last_call = nullptr;
  AnfNodePtr pre_last_call = nullptr;
  std::vector<FuncGraphManagerPtr> subgraph_managers;
  for (auto &kernel_cnode : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimCallInline)) {
      auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      auto mng = inline_subgraph->manager();
      if (mng == nullptr) {
        auto manager = MakeManager({inline_subgraph}, false);
        MS_EXCEPTION_IF_NULL(manager);
        manager->AddFuncGraph(inline_subgraph);
        inline_subgraph->set_manager(manager);
        // subgraph is not changed when InlineSubGraph, hold the manager of subgraph to avoid being released.
        subgraph_managers.emplace_back(manager);
      }
      InlineSubGraph(graph, inline_subgraph, kernel_cnode, &last_call, &pre_last_call, false);
    }
  }
  AscendGraphOptimization::GetInstance().OptimizeACLGraphAfterInline(graph);
  PROF_END(InlineCallGraph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

CNodePtr GetCondSwitchNode(const KernelGraphPtr &graph, const std::map<AnfNodePtr, size_t> &branch_input,
                           const AnfNodePtr &cond, std::map<AnfNodePtr, AnfNodePtr> *branch_tuple_getitem) {
  std::vector<AnfNodePtr> cond_switch_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionSwitch->name()))};
  cond_switch_inputs.resize(branch_input.size() + kIndex2);
  cond_switch_inputs[kIndex1] = cond;
  for (auto &kv : branch_input) {
    cond_switch_inputs[kv.second + kIndex2] = kv.first;
  }
  auto cond_switch_node = graph->NewCNode(cond_switch_inputs);
  MS_EXCEPTION_IF_NULL(cond_switch_node);

  for (auto &kv : branch_input) {
    if (branch_tuple_getitem->find(kv.first) == branch_tuple_getitem->end()) {
      auto tuple_getitem_node =
        graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), cond_switch_node, NewValueNode(SizeToLong(kv.second))});
      MS_EXCEPTION_IF_NULL(tuple_getitem_node);
      tuple_getitem_node->set_abstract(kv.first->abstract());
      (*branch_tuple_getitem)[kv.first] = tuple_getitem_node;
    }
  }
  AbstractBasePtrList abstract_list;
  for (size_t i = kIndex2; i < cond_switch_inputs.size(); ++i) {
    (void)abstract_list.emplace_back(cond_switch_inputs[i]->abstract());
  }
  cond_switch_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  SelectKernelInfo(graph, cond_switch_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(cond_switch_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  for (size_t input_index = 1; input_index < common::AnfAlgo::GetInputNum(cond_switch_node); ++input_index) {
    kernel_info->AddRefMap(input_index - 1, input_index);
  }
  return cond_switch_node;
}

CNodePtr GetBranchNode(const KernelGraphPtr &graph, const CNodePtr &old_branch_node,
                       const std::map<AnfNodePtr, AnfNodePtr> &branch_tuple_getitem) {
  std::vector<AnfNodePtr> branch_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartialInline->name()))};
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(old_branch_node); i++) {
    auto input = common::AnfAlgo::GetInputNode(old_branch_node, i);
    if (branch_tuple_getitem.find(input) != branch_tuple_getitem.end()) {
      branch_inputs.push_back(branch_tuple_getitem.at(input));
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, old_branch_node)
        << "Invalid input of branch node: " << old_branch_node->fullname_with_scope() << ", " << i << ", "
        << input->fullname_with_scope();
    }
  }
  auto branch_node = graph->NewCNode(branch_inputs);
  MS_EXCEPTION_IF_NULL(branch_node);
  branch_node->set_abstract(old_branch_node->abstract());
  SelectKernelInfo(graph, branch_node);
  common::AnfAlgo::CopyNodeAttrs(old_branch_node, branch_node);
  return branch_node;
}

void CollectInputByBranchCNode(const CNodePtr &true_branch_cnode, const CNodePtr &false_branch_cnode,
                               std::map<AnfNodePtr, size_t> *branch_input) {
  MS_EXCEPTION_IF_NULL(true_branch_cnode);
  MS_EXCEPTION_IF_NULL(false_branch_cnode);
  MS_EXCEPTION_IF_NULL(branch_input);
  std::set<AnfNodePtr> monad_inputs;
  auto now_input_cnt = 0;
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(true_branch_cnode); i++) {
    auto input = common::AnfAlgo::GetInputNode(true_branch_cnode, i);
    if (branch_input->find(input) != branch_input->end() || monad_inputs.find(input) != monad_inputs.end()) {
      continue;
    }
    if (HasAbstractMonad(input)) {
      monad_inputs.emplace(input);
      continue;
    }
    (*branch_input)[input] = now_input_cnt++;
  }
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(false_branch_cnode); i++) {
    auto input = common::AnfAlgo::GetInputNode(false_branch_cnode, i);
    if (branch_input->find(input) != branch_input->end() || monad_inputs.find(input) != monad_inputs.end()) {
      continue;
    }
    if (HasAbstractMonad(input)) {
      monad_inputs.emplace(input);
      continue;
    }
    (*branch_input)[input] = now_input_cnt++;
  }
  for (const auto &monad_input : monad_inputs) {
    (*branch_input)[monad_input] = now_input_cnt++;
  }
}

CNodePtr ProcessSwitchNode(const KernelGraphPtr &graph, const CNodePtr &kernel_cnode,
                           std::vector<CNodePtr> *partial_inline_cnode) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel_cnode);
  auto input_num = common::AnfAlgo::GetInputNum(kernel_cnode);
  MS_EXCEPTION_IF_CHECK_FAIL(input_num == kSwitchInputSize,
                             "Invalid input num of switch node: " + kernel_cnode->DebugString());
  auto cond = kernel_cnode->input(kSwitchCondIndex);
  auto true_branch = kernel_cnode->input(kSwitchBranchTrueIndex);
  auto false_branch = kernel_cnode->input(kSwitchBranchFalseIndex);
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(true_branch, prim::kPrimPartialInline),
                             "Invalid true branch of switch node: " + kernel_cnode->DebugString());
  MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(false_branch, prim::kPrimPartialInline),
                             "Invalid false branch of switch node: " + kernel_cnode->DebugString());
  auto true_branch_cnode = true_branch->cast<CNodePtr>();
  auto false_branch_cnode = false_branch->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(true_branch_cnode);
  MS_EXCEPTION_IF_NULL(false_branch_cnode);
  std::map<AnfNodePtr, size_t> branch_input;
  CollectInputByBranchCNode(true_branch_cnode, false_branch_cnode, &branch_input);
  std::map<AnfNodePtr, AnfNodePtr> branch_tuple_getitem;
  auto cond_switch_node = GetCondSwitchNode(graph, branch_input, cond, &branch_tuple_getitem);
  MS_EXCEPTION_IF_NULL(cond_switch_node);
  auto true_branch_node = GetBranchNode(graph, true_branch_cnode, branch_tuple_getitem);
  auto false_branch_node = GetBranchNode(graph, false_branch_cnode, branch_tuple_getitem);
  auto cond_gather_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionGather->name())), true_branch_node,
                     false_branch_node});
  cond_gather_node->set_abstract(kernel_cnode->abstract());
  SelectKernelInfo(graph, cond_gather_node);
  partial_inline_cnode->emplace_back(true_branch_node);
  partial_inline_cnode->emplace_back(false_branch_node);

  // Record the branch info for condition node.
  auto false_sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(false_branch_cnode, kAttrKernelGraph);
  auto true_sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(true_branch_cnode, kAttrKernelGraph);
  MS_EXCEPTION_IF_NULL(false_sub_graph);
  MS_EXCEPTION_IF_NULL(true_sub_graph);
  std::vector<ValuePtr> branch_graph_names;
  branch_graph_names.emplace_back(std::make_shared<StringImm>(false_sub_graph->ToString()));
  branch_graph_names.emplace_back(std::make_shared<StringImm>(true_sub_graph->ToString()));
  cond_switch_node->AddAttr(kInlineSubGraphName, std::make_shared<ValueTuple>(branch_graph_names));
  reverse(branch_graph_names.begin(), branch_graph_names.end());
  cond_gather_node->AddAttr(kAttrBranchGraphName, std::make_shared<ValueTuple>(branch_graph_names));
  graph->AddConditionGatherSwitchPair(cond_gather_node, cond_switch_node);
  MS_LOG(DEBUG) << "Add new condition gather node:" << cond_gather_node->fullname_with_scope()
                << " and condition switch actor:" << cond_switch_node->fullname_with_scope()
                << " for graph:" << graph->ToString();
  graph->FrontBackendlMapUpdate(kernel_cnode, cond_switch_node);
  MS_LOG(DEBUG) << "Update backend from:" << kernel_cnode->DebugString() << " to:" << cond_switch_node->DebugString()
                << " for front node:"
                << (graph->GetFrontAnfByBackendAnf(cond_switch_node) == nullptr
                      ? "null"
                      : graph->GetFrontAnfByBackendAnf(cond_switch_node)->DebugString());
  return cond_gather_node;
}

// Flatten the input abstract, and record the index construct.
// eg. tuple(tuple(1, 2), 3) -> {1, 2, 3}  ((-1, -1), -1)
AbstractBasePtrList CollectAbstract(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return {abstract};
  }
  const auto &seq_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    return {seq_abs};
  }

  AbstractBasePtrList abs_list;
  for (const auto &sub_abs : seq_abs->elements()) {
    AbstractBasePtrList sub_list = CollectAbstract(sub_abs);
    abs_list.insert(abs_list.end(), sub_list.begin(), sub_list.end());
  }
  return abs_list;
}

size_t ConstructAbstructIndex(const abstract::AbstractBasePtr &abstract, ValuePtr *abstract_construct_index) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    *abstract_construct_index = MakeValue(-1);
    return 1;
  }
  const auto &seq_abs = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    *abstract_construct_index = MakeValue(-1);
    return 1;
  }

  ValuePtrList construct_index_list;
  size_t index_num = 0;
  for (const auto &sub_abs : seq_abs->elements()) {
    ValuePtr sub_construct_index = nullptr;
    index_num += ConstructAbstructIndex(sub_abs, &sub_construct_index);
    construct_index_list.emplace_back(sub_construct_index);
  }
  *abstract_construct_index = std::make_shared<ValueTuple>(construct_index_list);
  return index_num;
}

// Rebuild the output construct by construct index.
CNodePtr ConstructMakeTupleRecursion(const ValuePtr &abstract_construct_index, std::deque<CNodePtr> *get_item_list,
                                     const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  MS_EXCEPTION_IF_NULL(get_item_list);
  MS_EXCEPTION_IF_NULL(graph);
  if (!abstract_construct_index->isa<ValueSequence>()) {
    if (get_item_list->empty()) {
      MS_LOG(EXCEPTION) << "Failed to get item node by value:" << abstract_construct_index->ToString();
    } else {
      auto top = get_item_list->front();
      get_item_list->pop_front();
      return top;
    }
  }

  // Build node and abstract for tuple construct.
  const auto &seq_value = abstract_construct_index->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_value);
  AnfNodePtrList node_list{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  AbstractBasePtrList abs_list;
  for (const auto &sub_value : seq_value->value()) {
    MS_EXCEPTION_IF_NULL(sub_value);
    const auto &new_node = ConstructMakeTupleRecursion(sub_value, get_item_list, graph);
    MS_EXCEPTION_IF_NULL(new_node);
    node_list.emplace_back(new_node);
    abs_list.emplace_back(new_node->abstract());
  }
  const auto &make_tuple = graph->NewCNode(node_list);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  return make_tuple;
}

AnfNodePtrList CreateTupleGetItemForTupleOutput(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract = node->abstract();
  if (abstract == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid abstract for node:" << node->DebugString();
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    return {node};
  }
  const auto &sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abstract);
  if (sequence_abstract->dynamic_len()) {
    return {node};
  }
  AnfNodePtrList outputs;
  for (size_t i = 0; i < sequence_abstract->elements().size(); ++i) {
    const auto &sub_abstract = sequence_abstract->elements()[i];
    MS_EXCEPTION_IF_NULL(sub_abstract);
    auto get_item = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), node,
                                     NewValueNode(MakeValue<int64_t>(SizeToLong(i)))});
    get_item->set_abstract(sub_abstract);
    const auto &sub_outputs = CreateTupleGetItemForTupleOutput(get_item, graph);
    outputs.insert(outputs.end(), sub_outputs.begin(), sub_outputs.end());
  }
  return outputs;
}

// Flatten the tuple input of condition gather.
CNodePtr FlattenConditionGatherNodeInput(const CNodePtr &kernel, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  if (mng == nullptr) {
    auto manager = MakeManager({graph});
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
    mng = graph->manager();
  }

  AnfNodePtrList new_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimConditionGather->name()))};
  size_t output_num = SIZE_MAX;
  // Collect inputs.
  for (size_t i = 1; i < kernel->inputs().size(); ++i) {
    const auto &input = kernel->inputs()[i];
    MS_EXCEPTION_IF_NULL(input);
    AnfNodePtrList outputs = CreateTupleGetItemForTupleOutput(input, graph);
    // All input branch should have same output num.
    if (output_num != SIZE_MAX && output_num != outputs.size()) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid output size:" << output_num << " and " << outputs.size()
                                          << " for kernel:" << kernel->fullname_with_scope();
    }
    output_num = outputs.size();
    new_inputs.insert(new_inputs.end(), outputs.begin(), outputs.end());
  }

  // Create new condition gather node.
  auto new_kernel = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(new_kernel);
  ValuePtr abstract_construct_index = nullptr;
  AbstractBasePtrList new_abstract_list = CollectAbstract(kernel->abstract());
  auto front_abs = kernel->abstract();
  auto gather_to_switch = graph->condition_gather_to_switch();
  const auto &switch_iter = gather_to_switch.find(kernel);
  if (switch_iter == gather_to_switch.end() || switch_iter->second == nullptr) {
    MS_LOG(WARNING) << "Failed to get condition switch node by condition gather:" << kernel->DebugString();
  } else {
    const auto &front_call = graph->GetFrontAnfByBackendAnf(switch_iter->second);
    if (front_call == nullptr || front_call->abstract() == nullptr) {
      MS_LOG(WARNING) << "Failed to get front call node by switch node:" << switch_iter->second->DebugString();
    } else {
      front_abs = front_call->abstract();
      MS_LOG(DEBUG) << "Rebuild output by front call node:" << front_call->DebugString()
                    << " abstract:" << front_abs->ToString()
                    << " by condition switch node:" << switch_iter->second->fullname_with_scope();
    }
  }
  size_t index_num = ConstructAbstructIndex(front_abs, &abstract_construct_index);
  MS_EXCEPTION_IF_NULL(abstract_construct_index);
  MS_LOG(INFO) << "Abstract construct index:" << abstract_construct_index->ToString()
               << " for rebuild the abstract of kernel:" << new_kernel->DebugString();
  if (new_abstract_list.size() != output_num || output_num != index_num) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid abstract list size:" << new_abstract_list.size()
                                        << " and output size:" << output_num << " output index size:" << index_num
                                        << " for kernel:" << kernel->DebugString()
                                        << " abstract:" << kernel->abstract()->ToString();
  }
  new_kernel->set_abstract(std::make_shared<abstract::AbstractTuple>(new_abstract_list));
  SelectKernelInfo(graph, new_kernel);
  if (output_num == SIZE_MAX) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Invalid output size:" << output_num
                                        << " for kernel:" << kernel->fullname_with_scope();
  }
  new_kernel->AddAttr(kAttrBranchOutputNum, MakeValue<size_t>(output_num));
  if (kernel->HasAttr(kAttrBranchGraphName)) {
    new_kernel->AddAttr(kAttrBranchGraphName, kernel->GetAttr(kAttrBranchGraphName));
  }

  // Rebuild the output construct for condition gather node.
  std::deque<CNodePtr> get_item_list;
  for (size_t i = 0; i < output_num; ++i) {
    auto get_item = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())),
                                     new_kernel, NewValueNode(MakeValue<int64_t>(i))});
    MS_EXCEPTION_IF_NULL(get_item);
    get_item_list.emplace_back(get_item);
    get_item->set_abstract(new_abstract_list[i]);
  }
  auto make_tuple = ConstructMakeTupleRecursion(abstract_construct_index, &get_item_list, graph);
  (void)mng->Replace(kernel, make_tuple);
  return new_kernel;
}

// Flatten the tuple input of condition node.
void FlattenConditionNodeInput(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_flatten_gather_input_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  const auto &nodes = TopoSort(graph->output());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto kernel = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(kernel);
    if (IsPrimitiveCNode(kernel, prim::kPrimConditionSwitch)) {
      for (size_t i = 2; i < kernel->size(); ++i) {
        const auto &input = kernel->input(i);
        MS_EXCEPTION_IF_NULL(input);
        if ((!input->isa<Parameter>()) || HasAbstractMonad(input) || common::AnfAlgo::HasAbstractRef(input)) {
          continue;
        }
        std::vector<AnfNodePtr> tensor_move_inputs = {
          NewValueNode(std::make_shared<Primitive>(prim::kPrimTensorMove->name())), input};
        auto tensor_move = graph->NewCNode(tensor_move_inputs);
        tensor_move->set_abstract(input->abstract()->Clone());
        kernel->set_input(i, tensor_move);
      }
    }
    if (!IsPrimitiveCNode(kernel, prim::kPrimConditionGather)) {
      continue;
    }
    const auto &new_kernel = FlattenConditionGatherNodeInput(kernel, graph);
    MS_EXCEPTION_IF_NULL(new_kernel);
    auto gather_to_switch = graph->condition_gather_to_switch();
    const auto &iter = gather_to_switch.find(kernel);
    if (iter == gather_to_switch.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Failed to get condition switch node for gather:" << kernel->DebugString();
    }
    MS_EXCEPTION_IF_NULL(iter->second);
    const auto &inline_iter = graph->inline_sub_graph_kernels().find(kernel);
    if (inline_iter != graph->inline_sub_graph_kernels().end()) {
      auto subgraph_name = inline_iter->second;
      graph->AddInlineSubgraphKernel(new_kernel, subgraph_name);
      MS_LOG(INFO) << "Add new condition gather node:" << new_kernel->fullname_with_scope()
                   << " subgraph name:" << subgraph_name << " to graph:" << graph->ToString();
    }
    graph->AddConditionGatherSwitchPair(new_kernel, iter->second);
    graph->RemoveConditionGatherSwitchPair(kernel);
    MS_LOG(INFO) << "Add new condition gather node:" << new_kernel->fullname_with_scope()
                 << " to replace node:" << kernel->fullname_with_scope() << " branch name:"
                 << (kernel->HasAttr(kAttrBranchGraphName) ? new_kernel->GetAttr(kAttrBranchGraphName)->ToString()
                                                           : " null")
                 << " in graph:" << graph->ToString();
  }

#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_flatten_gather_input_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

void InlineSwitchGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  PROF_START(InlineSwitchGraph);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    InlineSwitchGraph(child_graph.lock(), memo);
  }
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_switch_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  // process ConditionSwitch/ConditionGather
  graph->SetExecOrderByDefault();
  auto kernel_cnodes = graph->execution_order();
  auto mng = graph->manager();
  std::vector<CNodePtr> partial_inline_cnode;
  for (auto &kernel_cnode : kernel_cnodes) {
    if (!IsPrimitiveCNode(kernel_cnode, prim::kPrimSwitch)) {
      continue;
    }
    auto cond_gather_node = ProcessSwitchNode(graph, kernel_cnode, &partial_inline_cnode);
    if (mng == nullptr) {
      auto manager = MakeManager({graph});
      MS_EXCEPTION_IF_NULL(manager);
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
      mng = manager;
    }
    (void)mng->Replace(kernel_cnode, cond_gather_node);
  }

  // inline switch graph
  std::vector<FuncGraphManagerPtr> subgraph_managers;
  for (auto &kernel_cnode : partial_inline_cnode) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimPartialInline)) {
      auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      auto sub_mng = inline_subgraph->manager();
      if (sub_mng == nullptr) {
        auto sub_manager = MakeManager({inline_subgraph}, false);
        MS_EXCEPTION_IF_NULL(sub_manager);
        sub_manager->AddFuncGraph(inline_subgraph);
        inline_subgraph->set_manager(sub_manager);
        subgraph_managers.emplace_back(sub_manager);
      }
      InlineSubGraph(graph, inline_subgraph, kernel_cnode, nullptr, nullptr, true);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, kernel_cnode) << "Invalid node type, node: " << kernel_cnode->fullname_with_scope();
    }
  }
  FlattenConditionNodeInput(graph);
  PROF_END(InlineSwitchGraph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_switch_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

std::string GetBranchName(const KernelGraphPtr &graph, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(graph);
  std::string current_branch = graph->ToString();
  const auto &iter = graph->inline_sub_graph_kernels().find(kernel);
  if (iter != graph->inline_sub_graph_kernels().end()) {
    current_branch = iter->second;
  }
  return current_branch;
}

std::set<std::string> GetSwitchBranchName(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return {};
  }
  const auto &kernel = node->cast<CNodePtr>();
  if (kernel == nullptr || !common::AnfAlgo::CheckPrimitiveType(kernel, prim::kPrimConditionSwitch) ||
      !kernel->HasAttr(kInlineSubGraphName)) {
    return {};
  }
  const auto &inline_sub_graph_names = kernel->GetAttr(kInlineSubGraphName);
  if (inline_sub_graph_names == nullptr || !inline_sub_graph_names->isa<ValueTuple>()) {
    return {};
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::set<std::string> branch_names;
  for_each(tuple_name->value().begin(), tuple_name->value().end(),
           [&branch_names](const auto &value) { branch_names.emplace(GetValue<std::string>(value)); });
  return branch_names;
}

// Put the kernels belonging to the same inline subgraph together in the execution order.
void FixExecutionOrderForInlineControlFlowGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->condition_gather_to_switch().empty()) {
    return;
  }
  auto execution_order = graph->execution_order();
  for (const auto &condition_node_pair : graph->condition_gather_to_switch()) {
    std::vector<CNodePtr> new_order;
    std::vector<CNodePtr> new_order_after_switch;
    MS_EXCEPTION_IF_NULL(condition_node_pair.first);
    MS_EXCEPTION_IF_NULL(condition_node_pair.second);
    std::set<std::string> switch_branchs = GetSwitchBranchName(condition_node_pair.second);
    std::string current_branch = GetBranchName(graph, condition_node_pair.second->cast<CNodePtr>());
    bool is_get_switch = false;
    for (auto iter = execution_order.begin(); iter != execution_order.end(); ++iter) {
      if (*iter == condition_node_pair.second) {
        is_get_switch = true;
        continue;
      }
      if (*iter == condition_node_pair.first) {
        if (!is_get_switch) {
          MS_LOG_WITH_NODE(EXCEPTION, condition_node_pair.first)
            << "Condition gather:" << condition_node_pair.first->fullname_with_scope()
            << " is in front of condition switch: " << condition_node_pair.second->fullname_with_scope();
        }
        new_order.emplace_back(condition_node_pair.second->cast<CNodePtr>());
        new_order.insert(new_order.end(), new_order_after_switch.begin(), new_order_after_switch.end());
        new_order.insert(new_order.end(), iter, execution_order.end());
        break;
      }
      // In subgraph, the input of cnode maybe empty or all of them are value node, it maybe in front of the switch node
      // in toposort.
      if (!is_get_switch && switch_branchs.find(GetBranchName(graph, *iter)) != switch_branchs.end()) {
        MS_LOG(INFO) << "Kernel:" << (*iter)->fullname_with_scope() << " reorder behind the condition switch node.";
        new_order_after_switch.emplace_back(*iter);
      } else if (!is_get_switch || current_branch == GetBranchName(graph, *iter)) {
        new_order.emplace_back(*iter);
      } else {
        new_order_after_switch.emplace_back(*iter);
      }
    }
    if (execution_order.size() != new_order.size()) {
      MS_LOG(EXCEPTION) << "Failed to reorder execution kernel for graph:" << graph->ToString();
    }
    execution_order = new_order;
  }
  graph->set_execution_order(execution_order);
}
}  // namespace

void AscendKernelExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  device::ascend::AscendHalManager::GetInstance().InitializeAcl();
  MS_EXCEPTION_IF_NULL(device_context_);
  res_manager_ = device_context_->device_res_manager_.get();
  MS_EXCEPTION_IF_NULL(res_manager_);
  initialized_ = true;
}

void AscendKernelExecutor::Destroy() {
  if (!initialized_) {
    return;
  }
  res_manager_ = nullptr;
  initialized_ = false;
}

void AscendKernelExecutor::AddMindIRPass(const KernelGraphPtr &graph) const {
  AscendGraphOptimization::GetInstance().AscendUnifyMindIR(graph);
}

void AscendKernelExecutor::OptimizeGraph(const FuncGraphPtr &graph) const {
  // will be cached by OpCompileInfo
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  std::set<KernelGraphPtr> memo;
  AscendGraphOptimization::GetInstance().OptimizeACLGraph(kernel_graph, &memo);
  memo.clear();
  std::vector<size_t> op_selected_num(device::ascend::SelectedKernelType::NUM_KERNLE_TYPE, 0);
  bool has_aclop = false;
  SelectKernel(kernel_graph, &memo, &op_selected_num, &has_aclop);
  if (has_aclop) {
    static std::once_flag ge_init_flag_ = {};
    std::call_once(ge_init_flag_, [&]() {
      dynamic_cast<AscendDeviceContext *>(device_context_)->InitializeForAclop();
      SetAclOpPrecisionMode();
      res_manager_->SetAclDeterministic();
    });
  }
  PrintOpSelectedNum(op_selected_num);
  memo.clear();
  AscendGraphOptimization::GetInstance().OptimizeACLGraphAfterKernelSelect(kernel_graph, &memo);
  memo.clear();
  InlineCallGraph(kernel_graph);
  memo.clear();
  InlineSwitchGraph(kernel_graph, &memo);
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "GeOptimizeGraph", start_time,
                                  profiler::GetClockSyscnt(), 0);
}

void AscendKernelExecutor::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  if (nodes.empty()) {
    return;
  }
  auto func_graph = nodes[0]->func_graph();
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // build kernel mod
  MS_LOG(DEBUG) << "Status record: start create kernel.";
  uint64_t start_time = profiler::GetClockSyscnt();
  SetKernelInfoBeforeCreateKernel(nodes);
  auto ret = GenerateKernelMod(nodes);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Kernel build error.";
  }

  if (!kernel_graph->is_from_cache()) {
    AscendGraphOptimization::GetInstance().OptimizeACLGraphAfterCreateKernel(kernel_graph);
    OptimizeExecutionOrder(NOT_NULL(func_graph));
  } else {
    MS_LOG(INFO) << "Skip optimize after create kernel for:" << kernel_graph->ToString();
    ExecKernelModResize(kernel_graph);
    MS_LOG(INFO) << "Finish exec kernel mod resize for:" << kernel_graph->ToString();
  }
  (void)profiler::CollectHostInfo("Ascend", "CreateKernel", "CreateGeKernel", start_time, profiler::GetClockSyscnt(),
                                  1);
  MS_LOG(DEBUG) << "Status record: end create kernel.";
}

kernel::KernelModPtr AscendKernelExecutor::CreateKernelMod(const std::string &op_name) const {
  // Note: Only support generate aclnn kernel mod current.
  auto kernel_ptr = kernel::Factory<kernel::AclnnKernelMod>::Instance().Create(op_name);
  if (kernel_ptr == nullptr) {
    MS_LOG(WARNING) << "aclnn can't find Kernel[" << op_name << "]";
    return nullptr;
  }
  device::ascend::AclnnInit();
  return kernel_ptr;
}

void AscendKernelExecutor::DoStreamAssign(const KernelGraphPtr &kernel_graph) const {
  MS_LOG(DEBUG) << "Status record: start stream assign.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (ms_context->IsEnableInferBoost()) {
    MS_LOG(INFO) << "InferBoost mode, skip assign stream.";
    return;
  }
  MS_LOG(INFO) << "Status record: start stream assign, " << kernel_graph->ToString();
  // stream assign
  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeMultiStream)) {
    MS_LOG(INFO) << "Force single stream.";
  } else {
    AclStreamAssign::GetInstance().AssignStream(NOT_NULL(kernel_graph), res_manager_);
  }
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->CanDump(kIntroductory);
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_stream_assign_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  MS_LOG(INFO) << "Status record: end stream assign, " << kernel_graph->ToString();
}

void AscendKernelExecutor::DoSomas(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // somas
  MS_LOG(DEBUG) << "Status record: start do somas.";
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    auto somas = std::make_shared<AclSomas>();
    PROF_START(somas);
    bool ret = somas->Assign(kernel_graph);
    PROF_END(somas);
    if (ret) {
      MS_LOG(INFO) << "Somas allocate success for graph " << kernel_graph->graph_id()
                   << " somas size: " << kernel_graph->somas_whole_block_size();
    } else if (somas->IsSupportSomas(*kernel_graph)) {
      MS_LOG(WARNING) << "Somas allocate failed for graph " << kernel_graph->graph_id();
    }
  }
  MS_LOG(DEBUG) << "Status record: end do somas.";
}

void AscendKernelExecutor::OptimizeExecutionOrder(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  PROF_START(OptimizeExecutionOrder);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(DEBUG) << "Status record: start optimize execution order. graph id: " << kernel_graph->graph_id();
  kernel_graph->SetExecOrderByDefault();
  auto execution_order = kernel_graph->execution_order();
  kernel_graph->EnableRuntimeCache();
  common::AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  kernel_graph->DisableRuntimeCache();
  kernel_graph->set_execution_order(execution_order);
  MS_LOG(DEBUG) << "Status record: end optimize execution order. graph id: " << kernel_graph->graph_id();
  FixExecutionOrderForInlineControlFlowGraph(kernel_graph);
  PROF_END(OptimizeExecutionOrder);
}

namespace {
void CreateEventKernelMod(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = kernel_graph->execution_order();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsOneOfPrimitiveCNode(node, {prim::kPrimStreamSend, prim::kPrimStreamRecv, prim::kPrimResLimit})) {
      continue;
    }
    device::ascend::GenerateKernelBuildInfo(node, RT_KERNEL);
    auto kernel_mod_ptr = kernel::RtOpBuild(node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
  }
}
}  // namespace

void ResetCNodeName(const AnfNodePtrList &all_nodes) {
  mindspore::HashMap<std::string, int> node_ids;
  for (const auto &node : all_nodes) {
    if (node != nullptr && node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &fullname = cnode->fullname_with_scope();
      auto op_index = fullname.rfind("-op");
      if (op_index != string::npos) {
        auto scope_prefix = fullname.substr(0, op_index);
        if (node_ids.find(scope_prefix) == node_ids.end()) {
          node_ids[scope_prefix] = 0;
        } else {
          node_ids[scope_prefix]++;
        }
        cnode->set_fullname_with_scope(scope_prefix + "-op" + std::to_string(node_ids[scope_prefix]));
      }
    }
  }
}

void ResetNodeIds(const KernelGraphPtr &kernel_graph) {
  if (!kernel_graph->backend_jit_config().IsGptoOptionsEmpty()) {
    return;
  }
  if (!kernel_graph->memory_managed_by_ge()) {
    MS_LOG(INFO) << "Start reset node id";
    const auto &all_nodes = mindspore::TopoSort(kernel_graph->get_return(), SuccDeeperSimple);
    ResetCNodeName(all_nodes);
    MS_LOG(INFO) << "End reset node id";
  }
}

int AscendKernelExecutor::StressDetect(const std::string &detect_type) const {
  return kernel::pyboost::StressDetectKernel(detect_type);
}

int AscendKernelExecutor::CleanTdtChannel() const {
  if (!res_manager_->BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return 1;
  }
  MbufDataHandlerManager::GetInstance().CleanChannels();
  (void)device::DataQueueMgr::GetInstance().CleanTdtHandle();
  return 0;
}

void AscendKernelExecutor::PreprocessBeforeRun(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &nodes = kernel_graph->execution_order();
  const auto &all_nodes = TopoSort(graph->get_return());
  MS_VLOG(VL_FLOW) << "The size of execution order: " << nodes.size();
  MS_VLOG(VL_FLOW) << "The size of all node: " << all_nodes.size();
  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeCompileStat)) {
    std::cout << "The size of execution order: " << nodes.size() << std::endl;
    std::cout << "The size of all node: " << all_nodes.size() << std::endl;
  }

  // nop op -> memcpy
  for (const auto &node : nodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(node);
    // If the 2nd input of reshape is not a value node, then there are two inputs to select the host reshape operator
    bool is_host_reshape_op = false;
    if (op_name == prim::kPrimReshape->name()) {
      auto kernel_mod = AnfAlgo::GetKernelMod(node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      is_host_reshape_op = kernel_mod->GetKernelModType() == kernel::KernelModType::HostKernelMod;
    }
    bool is_nop_op = common::AnfAlgo::IsNopNode(node);
    bool is_transpose_nop =
      (op_name == prim::kPrimTransposeD->name()) && common::AnfAlgo::HasNodeAttr(kAttrNopOp, node);
    if (is_transpose_nop || (is_nop_op && !is_host_reshape_op)) {
      nop_op_to_memcpy_.insert(node);
    }
  }
  ResetNodeIds({kernel_graph});
  hyper_offload::ReorderControlFlowNodes(kernel_graph);
  hyper_offload::HyperOffloadOptimizer::Run(kernel_graph);
  DoStreamAssign(kernel_graph);
  hyper_offload::HyperOffloadOptimizer::AddEventToHyperOffloadOps(kernel_graph);
  CreateEventKernelMod(kernel_graph);
  ExecKernelModResize(kernel_graph);
  kernel_graph->PrintGraphExecuteOrder();
  DoSomas(NOT_NULL(graph));
  (void)profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", start_time,
                                  profiler::GetClockSyscnt(), 1);
}

void AscendKernelExecutor::CreateEventForCache(const KernelGraphPtr &kernel_graph) const {
  AclStreamAssign::GetInstance().CreateEvent(NOT_NULL(kernel_graph));
}

bool AscendKernelExecutor::MemoryCopyAsync(const CNodePtr &node, const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs, void *stream) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Launch MemoryCopyAsync instead for kernel " << node->fullname_with_scope();
  if (inputs.size() != 1 || outputs.size() != 1) {
    MS_LOG(DEBUG) << "Kernel " << node->fullname_with_scope() << " input output size should be 1 but"
                  << " input size is:" << inputs.size() << " output size is:" << outputs.size();
  }

  if (inputs[0]->size() == 0) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(stream);
  aclError status = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), outputs[0]->size(),
                                    inputs[0]->device_ptr(), inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "MemCpyAsync op aclrtMemcpyAsync failed, ret:" << status << " destMax:" << outputs[0]->size()
                  << " count:" << inputs[0]->size();
    return false;
  }
  return true;
}

bool AscendKernelExecutor::LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                        void *stream) const {
  // launch kernel
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  bool ret = true;
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    ret = MemoryCopyAsync(kernel, inputs, outputs, stream);
    if (!ret) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
    }
  } else {
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_EXCEPTION_IF_NULL(stream);
    ret = kernel_mod->Launch(inputs, workspace, outputs, stream);

    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
    }
  }

  if (!ret) {
    return false;
  }
  // for PyNative and KBK Sync Run mode
  if (MS_UNLIKELY(mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking())) {
    if (!AscendStreamMng::GetInstance().SyncStream(stream)) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel)
        << "Sync run failed, detail: " << CALL_ASCEND_API(aclGetRecentErrMsg) << trace::DumpSourceLines(kernel);
    }
  }
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
               kernel->fullname_with_scope(), false, inputs);
  return true;
}

bool AscendKernelExecutor::LaunchKernelHP(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod,
                                          void *stream) const {
  if (nop_op_to_memcpy_.find(kernel) != nop_op_to_memcpy_.end()) {
    if (!MemoryCopyAsync(kernel, inputs, outputs, stream)) {
      MS_LOG(ERROR) << "Memory copy failed for kernel " << kernel->fullname_with_scope();
      return false;
    }
  } else {
    bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
      res_manager_->ResetStreamAndCtx();
      return false;
    }
  }
  return true;
}

void AclrtLaunchCallback(void *user_data) {
  CallbackFunc *callback_func = reinterpret_cast<CallbackFunc *>(user_data);
  (*callback_func)();
  delete callback_func;
}

bool AscendKernelExecutor::LaunchCallback(CallbackFunc callback_func, size_t stream_id, bool is_block) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream_id = kDefaultStreamIndex;
    stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  }
  MS_EXCEPTION_IF_NULL(stream);
  auto block_type =
    is_block ? aclrtCallbackBlockType::ACL_CALLBACK_BLOCK : aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK;
  auto callback_func_ptr = new CallbackFunc(callback_func);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callback_func_ptr, block_type, stream);
  MS_LOG(DEBUG) << "Launch callback for stream_id : " << stream_id << ", ret : " << ret << ".";
  if (ret) {
    delete callback_func_ptr;
    MS_LOG(ERROR) << "Launch callback for stream_id : " << stream_id << " failed, ret : " << ret << ".";
    if (res_manager_->SyncStream(stream_id)) {
      callback_func();
      return true;
    }
    auto arf_env = common::GetEnv("MS_ENABLE_TFT");
    constexpr std::string_view arf = "ARF:1";
    if (arf_env.find(arf) == std::string::npos) {
      res_manager_->ResetStreamAndCtx();
    }
    return false;
  }
  return true;
}

void CustomizeCopyAscendInner(device::DeviceContext *device_context, const tensor::TensorPtr &input_tensor,
                              const tensor::TensorPtr &output_tensor, const size_t &stream_id) {
  const auto &input_addr = input_tensor->device_address();
  MS_EXCEPTION_IF_NULL(input_addr);
  const auto &input_storage_info = input_tensor->storage_info();

  const auto &output_addr = output_tensor->device_address();
  MS_EXCEPTION_IF_NULL(output_addr);
  const auto &output_storage_info = output_tensor->storage_info();

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", device::tracker::MemType::kPyNativeOutput,
                                                 output_addr->GetSize(), output_addr.get());
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
    MarkTensorAsInput, "PyNative", device::GetDeviceNameByType(input_addr->GetDeviceType()), input_addr->GetPtr(),
    input_tensor->data_type(), input_tensor->shape(), input_storage_info);
  if (output_addr->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(output_addr.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", output_storage_info:" << (output_storage_info == nullptr ? "" : output_storage_info->ToString())
                << ", input address size:" << input_addr->GetSize()
                << ", output address size:" << output_addr->GetSize();

  // Inplace output need be front
  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, stream_id, output_tensor, input_tensor);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
    MarkTensorAsOutput, "PyNative", device::GetDeviceNameByType(output_addr->GetDeviceType()), output_addr->GetPtr(),
    output_tensor->data_type(), output_tensor->shape(), output_storage_info);
  MS_LOG(DEBUG) << "Launch end";
}

// Unconventional pyboost writing. Please do not refer to this to implement other operators!
void CustomizeCopyAscend(device::DeviceContext *device_context, const tensor::TensorPtr &input_tensor,
                         const tensor::TensorPtr &output_tensor, const size_t &stream_id) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(output_tensor);

  if (runtime::Pipeline::Get().backend_stage()->CanPush()) {
    MS_LOG(DEBUG) << "Dispatch inplacecopy to backend queue";
    kernel::pyboost::PyBoostUtils::DispatchRun(
      std::make_shared<runtime::PyBoostDeviceTask>([device_context, input_tensor, output_tensor, stream_id]() {
        CustomizeCopyAscendInner(device_context, input_tensor, output_tensor, stream_id);
      }));
    return;
  }

  runtime::Pipeline::Get().WaitForward();
  CustomizeCopyAscendInner(device_context, input_tensor, output_tensor, stream_id);
  MS_LOG(DEBUG) << "Launch end";
}

bool AscendKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                             const tensor::TensorPtrList &input_tensors,
                                             const tensor::TensorPtrList &output_tensors,
                                             const size_t &stream_id) const {
  MS_LOG(DEBUG) << "task_type:" << task_type;
  if (runtime::KernelTaskType::kCOPY_TASK == task_type) {
    constexpr size_t kCopyTaskInputsNum = 2;
    // Copy task is a in-place op, the output is the first input.
    // To reuse the aclnnInplaceCopy, the first input of Copy is used as the operator output,
    // and the second input is used as the operator input.
    if (input_tensors.size() != kCopyTaskInputsNum) {
      MS_LOG(EXCEPTION) << "input_tensors.size() is invalid, input_tensors.size():" << input_tensors.size();
    }
    CustomizeCopyAscend(device_context_, input_tensors[1], input_tensors[0], stream_id);
  } else {
    // For contiguous task, there must be at least one input and one output.
    if (input_tensors.empty() || output_tensors.empty()) {
      MS_LOG(EXCEPTION) << "input_tensors.size() or output_tensors.size() is invalid, input_tensors.size():"
                        << input_tensors.size() << ", output_tensors.size():" << output_tensors.size();
    }

    const auto &input = input_tensors[0];
    const auto &input_device_address = std::static_pointer_cast<device::DeviceAddress>((input->device_address()));
    MS_EXCEPTION_IF_NULL(input_device_address);
    const auto &input_storage_info = input->storage_info();

    const auto &output = output_tensors[0];
    const auto &output_device_address = std::static_pointer_cast<device::DeviceAddress>(output->device_address());
    MS_EXCEPTION_IF_NULL(output_device_address);

    if (input_device_address != nullptr && input_device_address->GetDeviceType() == device::DeviceType::kCPU &&
        output_device_address != nullptr && output_device_address->GetDeviceType() == device::DeviceType::kCPU) {
      // for unrefmode, the output is on host, just copy the device ptr
      output_device_address->set_ptr(input_device_address->GetMutablePtr());
    } else {
      CustomizeCopyAscend(device_context_, input_tensors[0], output_tensors[0], stream_id);
    }
  }

  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  // for PyNative and KBK Sync Run mode
  bool ret = true;
  if (mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking()) {
    ret = AscendStreamMng::GetInstance().SyncStream(stream);
  }
  return ret;
}

// Task for graph mode, which receive pointers as input, it is used for convert input contiguous by aclnnInplaceCopy.
bool AscendKernelExecutor::ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                                             const std::vector<KernelTensor *> &input_kernel_tensors,
                                             const std::vector<KernelTensor *> &output_kernel_tensors,
                                             const size_t &stream_id) const {
  MS_LOG(DEBUG) << "task_type:" << task_type;
  MS_LOG(DEBUG) << "Graph call contiguous start";
  const auto &input = input_kernel_tensors[0];
  auto input_addr = input->device_address();
  MS_EXCEPTION_IF_NULL(input_addr);

  const auto &output = output_kernel_tensors[0];
  auto output_addr = output->device_address();
  MS_EXCEPTION_IF_NULL(output_addr);

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "Graph", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "Graph", device::tracker::MemType::kPyNativeOutput,
                                                 output_addr->GetSize(), output_addr.get());

  if (output_addr->GetPtr() == nullptr) {
    if (!device_context_->device_res_manager_->AllocateMemory(output_addr.get())) {
      MS_LOG(EXCEPTION) << "Allocate memory failed";
    }
  }

  const auto &input_storage_info = input->tensor_storage_info();
  const auto &output_storage_info = output->tensor_storage_info();
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", output_storage_info:" << (output_storage_info == nullptr ? "" : output_storage_info->ToString())
                << ", input address size:" << input_addr->GetSize()
                << ", output address size:" << output_addr->GetSize();

  auto stream_ptr = device_context_->device_res_manager_->GetStream(stream_id);
  auto res = GEN_EXECUTOR(std::string("aclnnInplaceCopy"), output, input);
  auto workspace_size = std::get<0>(res);
  auto executor = std::get<1>(res);
  auto release_func = std::get<kIndex3>(res);
  if (workspace_size == 0) {
    RUN_OP_API_ASYNC(std::string("aclnnInplaceCopy"), nullptr, 0, executor, stream_ptr, release_func);
  } else {
    auto workspace_addr = device_context_->device_res_manager_->AllocateMemory(workspace_size);
    RUN_OP_API_ASYNC(std::string("aclnnInplaceCopy"), workspace_addr, workspace_size, executor, stream_ptr,
                     release_func);
    device_context_->device_res_manager_->FreeMemory(workspace_addr);
  }
  MS_LOG(DEBUG) << "Graph call contiguous end";
  return true;
}
}  // namespace mindspore::device::ascend
